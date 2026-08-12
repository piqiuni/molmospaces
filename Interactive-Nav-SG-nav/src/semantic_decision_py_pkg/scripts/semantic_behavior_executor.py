#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque

import base64
import cv2
import numpy as np
from semantic_decision_py_pkg.behavior_execution import (
    BehaviorExecutionStateMachine,
    ExecutionConfig,
    NavigationProgressWatchdog,
    PostInteractionCostmapBaseline,
    PostInteractionPlanningMapBarrier,
    PostInteractionRawMapBarrier,
    STATE_APPROACH_INTERACTION,
    STATE_IDLE,
    STATE_INTERACTING,
    STATE_NAVIGATING,
    STATE_PREPARING_EXPLORE,
    STATE_SCANNING,
    STATE_WAITING_FOR_DRAWER_SCAN,
    STATE_WAITING_FOR_INTERACTION_OBSERVATION,
    STATE_VERIFYING,
    bounded_empty_plan_retry_delay,
    candidate_with_effective_interaction_approach,
    committed_turn_sign,
    container_two_stage_action_goal_options_for_staging,
    interaction_pose_validation,
    is_container_two_stage_physical_action,
    is_post_interaction_traversal_navigation,
    is_interaction_pose_precondition_failure,
    navigation_goal_options,
    navigation_prerotation_heading_target,
    navigation_requires_final_yaw,
    navigation_should_prerotate,
    next_interaction_approach_option_index,
    normalize_angle,
    path_lookahead_point,
    post_open_path_is_confirmed,
    post_open_path_retryable_preflight_reason,
    post_interaction_costmap_baseline_keys,
    post_interaction_costmap_receipts_fresh_source,
    post_interaction_planning_occupancy_fresh_source,
    post_interaction_raw_occupancy_fresh_source,
    prerotation_control_step_budget,
    requires_graph_verification,
    is_static_portal_interaction_feedback,
    is_stuck_recovery_failure,
    safe_grid_motion_distance,
)
from semantic_decision_py_pkg.ros_compat import patch_roslogging_findcaller_for_py311
from semantic_decision_py_pkg.step_command_gate import StepCommandGate
from semantic_decision_py_pkg.external_recovery import (
    normalize_recovery_request,
    recovery_request_matches_selection,
)
from semantic_decision_py_pkg.startup_scan_timing import (
    startup_scan_elapsed_control_s,
    startup_scan_timeout_reason,
)
from semantic_decision_py_pkg.visual_interaction_planning import (
    action_for_opaque_open_contract,
    candidate_with_direct_drawer_scan,
    candidate_with_visual_drawer_scan,
    candidate_with_visual_operation_plan,
    fresh_direct_drawer_scan_candidate,
    infer_visual_interaction_target_type,
)
from semantic_mllm_py_pkg.ablation import AblationConfig
from semantic_mllm_py_pkg.client import MLLMClient
from semantic_mllm_py_pkg.env import client_config_from_env, load_env_file
from semantic_mllm_py_pkg.interaction_prompt import (
    VISUAL_INTERACTION_PLANNING_INSTRUCTION,
    visual_interaction_planning_context,
)
from semantic_mllm_py_pkg.schemas import (
    build_visual_interaction_plan_response_schema,
    build_visual_verification_response_schema,
    validate_visual_interaction_plan,
    validate_visual_verification,
)

patch_roslogging_findcaller_for_py311()

import actionlib
import rospy
import tf
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from map_msgs.msg import OccupancyGridUpdate
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.srv import GetPlan
from sensor_msgs.msg import Image
from std_msgs.msg import String


TERMINAL_STATES = {
    GoalStatus.PREEMPTED,
    GoalStatus.SUCCEEDED,
    GoalStatus.ABORTED,
    GoalStatus.REJECTED,
    GoalStatus.RECALLED,
    GoalStatus.LOST,
}


def _rotation_arc_for_sign(
    current_yaw: float, target_yaw: float, turn_sign: int
) -> float:
    """Return the non-negative CW/CCW arc from ``current`` to ``target``.

    ``angular.z > 0`` is counter-clockwise.  We keep the two arcs separate
    instead of normalising to the shortest one because a collision-safe turn
    may have to use the non-shortest side.  The caller still bounds the arc by
    its finite evaluator-step budget.
    """

    full_turn = 2.0 * math.pi
    if int(turn_sign) >= 0:
        return float((float(target_yaw) - float(current_yaw)) % full_turn)
    return float((float(current_yaw) - float(target_yaw)) % full_turn)


def _costmap_origin_yaw(occupancy: OccupancyGrid) -> float:
    orientation = occupancy.info.origin.orientation
    return math.atan2(
        2.0
        * (
            float(orientation.w) * float(orientation.z)
            + float(orientation.x) * float(orientation.y)
        ),
        1.0
        - 2.0
        * (
            float(orientation.y) * float(orientation.y)
            + float(orientation.z) * float(orientation.z)
        ),
    )


def circular_costmap_footprint_is_clear(
    data: list[int] | tuple[int, ...],
    width: int,
    height: int,
    resolution: float,
    origin_xy: tuple[float, float],
    origin_yaw: float,
    center_xy: tuple[float, float],
    robot_radius_m: float,
    safety_margin_m: float,
    *,
    occupied_threshold: int = 50,
    unknown_is_blocked: bool = True,
) -> bool:
    """Conservatively check the configured circular local-costmap footprint.

    The running stack exposes ``robot_radius`` (0.25 m), not a polygonal
    footprint.  A disc is therefore the only public footprint contract that
    can be evaluated consistently by the executor.  Unknown/out-of-map cells
    are intentionally rejected for direct bridge control.
    """

    width = int(width)
    height = int(height)
    resolution = float(resolution)
    clearance = max(0.0, float(robot_radius_m) + float(safety_margin_m))
    if width <= 0 or height <= 0 or resolution <= 0.0:
        return False
    if len(data) < width * height:
        return False
    origin_x, origin_y = float(origin_xy[0]), float(origin_xy[1])
    center_x, center_y = float(center_xy[0]), float(center_xy[1])
    cosine = math.cos(float(origin_yaw))
    sine = math.sin(float(origin_yaw))

    # Convert the world-space centre to the grid's possibly rotated frame.
    delta_x = center_x - origin_x
    delta_y = center_y - origin_y
    local_center_x = cosine * delta_x + sine * delta_y
    local_center_y = -sine * delta_x + cosine * delta_y
    center_col = int(math.floor(local_center_x / resolution))
    center_row = int(math.floor(local_center_y / resolution))
    footprint_cells = int(math.ceil(clearance / resolution))
    for row in range(center_row - footprint_cells, center_row + footprint_cells + 1):
        for col in range(center_col - footprint_cells, center_col + footprint_cells + 1):
            local_cell_x = (float(col) + 0.5) * resolution
            local_cell_y = (float(row) + 0.5) * resolution
            if math.hypot(local_cell_x - local_center_x, local_cell_y - local_center_y) > clearance:
                continue
            if col < 0 or row < 0 or col >= width or row >= height:
                return False
            value = int(data[row * width + col])
            if value >= int(occupied_threshold) or (
                bool(unknown_is_blocked) and value < 0
            ):
                return False
    return True


def circular_costmap_rotation_sweep_is_clear(
    data: list[int] | tuple[int, ...],
    width: int,
    height: int,
    resolution: float,
    origin_xy: tuple[float, float],
    origin_yaw: float,
    center_xy: tuple[float, float],
    rotation_arc_rad: float,
    sweep_step_rad: float,
    robot_radius_m: float,
    safety_margin_m: float,
    *,
    occupied_threshold: int = 50,
    unknown_is_blocked: bool = True,
) -> bool:
    """Check every sampled pose of a bounded in-place rotation sweep.

    With the currently configured circular footprint the occupied cells are
    orientation invariant, but we still iterate the requested angular sweep.
    That keeps the control contract explicit and makes the method safe to
    replace with a public polygon footprint later without changing callers.
    """

    arc = max(0.0, float(rotation_arc_rad))
    step = max(1e-3, float(sweep_step_rad))
    sample_count = max(1, int(math.ceil(arc / step)))
    for _sample_index in range(sample_count + 1):
        if not circular_costmap_footprint_is_clear(
            data,
            width,
            height,
            resolution,
            origin_xy,
            origin_yaw,
            center_xy,
            robot_radius_m,
            safety_margin_m,
            occupied_threshold=occupied_threshold,
            unknown_is_blocked=unknown_is_blocked,
        ):
            return False
    return True


def circular_costmap_linear_sweep_distance(
    data: list[int] | tuple[int, ...],
    width: int,
    height: int,
    resolution: float,
    origin_xy: tuple[float, float],
    origin_yaw: float,
    start_xyyaw: tuple[float, float, float],
    direction_sign: float,
    requested_distance_m: float,
    robot_radius_m: float,
    safety_margin_m: float,
    *,
    occupied_threshold: int = 50,
    unknown_is_blocked: bool = True,
) -> float:
    """Return the collision-checked distance of a straight local backoff."""

    requested = max(0.0, float(requested_distance_m))
    resolution = float(resolution)
    if requested <= 0.0 or resolution <= 0.0:
        return 0.0
    start_x, start_y, yaw = (float(value) for value in start_xyyaw)
    if not circular_costmap_footprint_is_clear(
        data,
        width,
        height,
        resolution,
        origin_xy,
        origin_yaw,
        (start_x, start_y),
        robot_radius_m,
        safety_margin_m,
        occupied_threshold=occupied_threshold,
        unknown_is_blocked=unknown_is_blocked,
    ):
        return 0.0
    sample_step = max(0.02, 0.5 * resolution)
    direction = 1.0 if float(direction_sign) >= 0.0 else -1.0
    safe_distance = 0.0
    distance = min(sample_step, requested)
    while distance <= requested + 1e-9:
        center = (
            start_x + direction * distance * math.cos(yaw),
            start_y + direction * distance * math.sin(yaw),
        )
        if not circular_costmap_footprint_is_clear(
            data,
            width,
            height,
            resolution,
            origin_xy,
            origin_yaw,
            center,
            robot_radius_m,
            safety_margin_m,
            occupied_threshold=occupied_threshold,
            unknown_is_blocked=unknown_is_blocked,
        ):
            break
        safe_distance = min(distance, requested)
        if safe_distance >= requested:
            break
        distance = min(requested, distance + sample_step)
    return safe_distance


def rear_dwa_oscillation_detected(
    samples: list[dict] | tuple[dict, ...],
    *,
    minimum_samples: int,
    minimum_sign_flips: int,
    displacement_m: float,
    maximum_displacement_m: float,
    goal_distance_reduction_m: float,
    minimum_goal_distance_reduction_m: float,
) -> bool:
    """Detect left/right DWA churn only when it has made no real progress."""

    relevant = [
        sample
        for sample in samples
        if abs(float(sample.get("angular_z", 0.0))) > 1e-6
    ]
    if len(relevant) < max(2, int(minimum_samples)):
        return False
    signs = [1 if float(sample["angular_z"]) > 0.0 else -1 for sample in relevant]
    sign_flips = sum(
        1 for previous, current in zip(signs, signs[1:]) if previous != current
    )
    return bool(
        sign_flips >= max(1, int(minimum_sign_flips))
        and float(displacement_m) < max(0.0, float(maximum_displacement_m))
        and float(goal_distance_reduction_m)
        < max(0.0, float(minimum_goal_distance_reduction_m))
    )


class SemanticBehaviorExecutor:
    def __init__(self) -> None:
        env_path = os.environ.get("SEMANTIC_DECISION_ENV_FILE")
        # Keep an explicitly selected endpoint and its credential paired even
        # when the ROS launcher inherited a different OPENAI_API_KEY.
        load_env_file(env_path, override=bool(env_path))
        rospy.init_node("semantic_behavior_executor")
        topics = rospy.get_param("~topics", {}) or {}
        config = rospy.get_param("~executor", {}) or {}
        ablation_config = rospy.get_param("~ablation", {}) or {}
        self.ablation = AblationConfig(
            module1=str(ablation_config.get("module1", "dynamic_rule")),
            module2=str(ablation_config.get("module2", "rule_cost")),
            module3=str(ablation_config.get("module3", "rule_verified")),
        )
        model_config = rospy.get_param("~model", {}) or {}
        # The explicitly selected semantic-model env file is the cross-module
        # source of truth.  It must override a legacy nonempty YAML model name
        # so Module 3 follows the same local endpoint/model as Modules 1 and 2.
        model_name = str(
            os.environ.get("SEMANTIC_MODEL_NAME")
            or model_config.get("model", "")
            or ""
        )
        self.mllm_client = MLLMClient(
            client_config_from_env(
                model=model_name or None,
                metrics_path=str(model_config.get("metrics_path", "") or "") or None,
            )
        )
        self.skill_max_output_tokens = max(
            128,
            int(model_config.get("skill_max_output_tokens", min(self.mllm_client.config.max_tokens, 256))),
        )
        self.verification_max_output_tokens = max(
            128,
            int(model_config.get("verification_max_output_tokens", min(self.mllm_client.config.max_tokens, 256))),
        )
        self.skill_timeout_s = max(
            0.1, float(model_config.get("skill_timeout_s", 4.0))
        )
        self.verification_timeout_s = max(
            0.1, float(model_config.get("verification_timeout_s", 4.0))
        )
        # M3 is now an audit after the sealed backend result.  Keep one short
        # wait for a post-action image and at most one M1 re-observation when
        # it disagrees; neither setting is allowed to stall execution.
        self.post_interaction_visual_audit_wait_s = max(
            0.0, float(config.get("post_interaction_visual_audit_wait_s", 2.0))
        )
        self.post_interaction_visual_audit_max_reobserves = max(
            0, int(config.get("post_interaction_visual_audit_max_reobserves", 1))
        )
        self.mllm_crop_margin_ratio = max(
            0.0, float(model_config.get("crop_margin_ratio", 0.10))
        )
        self.mllm_crop_max_side_px = max(
            128, int(model_config.get("crop_max_side_px", 512))
        )
        self.mllm_full_image_max_side_px = max(
            256, int(model_config.get("full_image_max_side_px", 960))
        )
        self.drawer_scan_wait_timeout_s = max(
            0.1, float(config.get("drawer_scan_wait_timeout_s", 8.0))
        )
        self.drawer_scan_wait_poll_interval_s = max(
            0.01, float(config.get("drawer_scan_wait_poll_interval_s", 0.10))
        )
        # A grounded drawer scan is a finite simulator macro (open -> observe
        # -> close per selected front, then restore the view).  Its completion
        # must not depend on host/render wall time.  Keep a hard step budget and
        # a no-step liveness guard; ordinary portal/fridge interactions retain
        # the generic wall-clock timeout below.
        self.drawer_scan_execution_step_budget_authoritative = bool(
            config.get("drawer_scan_execution_step_budget_authoritative", True)
        )
        self.drawer_scan_execution_max_task_steps = max(
            1, int(config.get("drawer_scan_execution_max_task_steps", 120))
        )
        self.drawer_scan_execution_step_budget_margin = max(
            0, int(config.get("drawer_scan_execution_step_budget_margin", 8))
        )
        self.drawer_scan_execution_step_sync_stall_timeout_s = max(
            0.1,
            float(
                config.get(
                    "drawer_scan_execution_step_sync_stall_timeout_s", 5.0
                )
            ),
        )
        self.drawer_scan_execution_wall_cap_s = max(
            1.0, float(config.get("drawer_scan_execution_wall_cap_s", 180.0))
        )
        # A targeted M1 refresh is an observation barrier, not a simulator
        # motion timeout.  It waits for a later RGB+detection pair and then
        # consumes the public attribute patch; physical actions remain step
        # gated elsewhere.
        self.interaction_observation_timeout_s = max(
            0.1, float(config.get("interaction_observation_timeout_s", 30.0))
        )
        self.interaction_observation_poll_interval_s = max(
            0.01, float(config.get("interaction_observation_poll_interval_s", 0.05))
        )
        # The physical action is already separated from perception: M1 runs at
        # a farther outer staging pose, while the bridge later validates the
        # nearer pose and refrigerator sweep privately.  One fresh *direct*
        # front observation is therefore the useful default; requiring two
        # made harmless front/oblique model flicker suppress every action.
        # Deployments can still opt into a temporal vote through config.
        self.container_pre_action_confirmation_count = max(
            1, int(config.get("container_pre_action_confirmation_count", 1))
        )
        self.container_pre_action_require_direct_front = bool(
            config.get("container_pre_action_require_direct_front", True)
        )
        # Targeted M1 evidence is valid only at the staging pose that produced
        # it.  The pose check is intentionally tighter than the bridge's broad
        # arrival tolerance so a close/oblique overshoot cannot be mistaken for
        # a fresh front observation.
        self.container_m1_capture_pose_tolerance_m = max(
            0.05,
            float(config.get("container_m1_capture_pose_tolerance_m", 0.18)),
        )
        self.container_m1_capture_yaw_tolerance_rad = max(
            0.05,
            float(config.get("container_m1_capture_yaw_tolerance_rad", 0.25)),
        )
        # A transient M1 flip at one fixed staging pose should consume another
        # observation, not discard the recently valid front/action-region
        # evidence and jump to a different side.  This is a count bound, not a
        # wall-clock timeout.
        self.container_m1_same_pose_flip_retry_count = max(
            0,
            int(config.get("container_m1_same_pose_flip_retry_count", 1)),
        )
        startup_scan_config = rospy.get_param("~startup_scan", {}) or {}
        self.startup_scan_enabled = bool(startup_scan_config.get("enabled", False))
        self.startup_scan_angle_rad = max(
            0.0, float(startup_scan_config.get("angle_rad", 2.0 * math.pi))
        )
        self.startup_scan_angular_speed_rad_s = float(
            startup_scan_config.get("angular_speed_rad_s", 1.25)
        )
        self.startup_scan_control_dt_s = max(
            1e-3, float(startup_scan_config.get("control_dt_s", 0.2))
        )
        self.startup_scan_timeout_s = max(
            0.0, float(startup_scan_config.get("timeout_s", 15.0))
        )
        self.startup_scan_max_control_steps = max(
            1, int(startup_scan_config.get("max_control_steps", 40))
        )
        self.startup_scan_max_pending_steps = max(
            2, int(startup_scan_config.get("max_pending_steps", 32))
        )
        self.startup_scan_pair_max_age_s = max(
            0.0, float(startup_scan_config.get("pair_max_age_s", 0.75))
        )
        self.startup_scan_step_sync_stall_timeout_s = max(
            0.1,
            float(startup_scan_config.get("step_sync_stall_timeout_s", 5.0)),
        )
        self.machine = BehaviorExecutionStateMachine(
            ExecutionConfig(
                navigation_timeout_s=float(config.get("navigation_timeout_s", 180.0)),
                interaction_navigation_timeout_s=float(
                    config.get("interaction_navigation_timeout_s", 180.0)
                ),
                interaction_timeout_s=float(config.get("interaction_timeout_s", 30.0)),
                drawer_scan_wait_timeout_s=self.drawer_scan_wait_timeout_s,
                interaction_observation_timeout_s=self.interaction_observation_timeout_s,
                verification_timeout_s=float(config.get("verification_timeout_s", 30.0)),
                explore_prepare_timeout_s=float(
                    config.get("explore_prepare_timeout_s", 10.0)
                ),
                explore_finalize_timeout_s=float(
                    config.get("explore_finalize_timeout_s", 10.0)
                ),
                # Startup scan timeout is evaluator control time, handled in
                # _run_startup_scan.  Disable the generic wall-clock guard.
                scan_timeout_s=0.0,
            )
        )
        self.map_frame = str(config.get("map_frame", "tf_frame_map"))
        self.base_frame = str(config.get("base_frame", "tf_frame_base_link"))
        self.rear_goal_prerotate_enabled = bool(
            config.get("rear_goal_prerotate_enabled", True)
        )
        self.rear_goal_enter_angle_rad = min(
            math.pi,
            max(
                0.0,
                float(config.get("rear_goal_enter_angle_rad", math.pi / 2.0)),
            ),
        )
        self.rear_goal_exit_angle_rad = float(
            config.get("rear_goal_exit_angle_rad", 0.34)
        )
        self.rear_goal_rotate_speed_rad_s = float(
            config.get("rear_goal_rotate_speed_rad_s", 1.25)
        )
        self.rear_goal_prerotate_timeout_s = float(
            config.get("rear_goal_prerotate_timeout_s", 12.0)
        )
        self.rear_goal_prerotate_step_sync_enabled = bool(
            config.get("rear_goal_prerotate_step_sync_enabled", True)
        )
        self.rear_goal_prerotate_control_dt_s = max(
            1e-3,
            float(config.get("rear_goal_prerotate_control_dt_s", 0.2)),
        )
        self.rear_goal_prerotate_max_control_steps = max(
            1,
            int(config.get("rear_goal_prerotate_max_control_steps", 28)),
        )
        self.rear_goal_prerotate_post_budget_settle_steps = max(
            0,
            int(config.get("rear_goal_prerotate_post_budget_settle_steps", 3)),
        )
        self.rear_goal_prerotate_step_sync_stall_timeout_s = max(
            0.1,
            float(config.get("rear_goal_prerotate_step_sync_stall_timeout_s", 2.0)),
        )
        # Rear-goal pre-rotation is an evaluator-synchronous cmd_vel behavior,
        # like the mandatory startup scan.  Keep its event buffer independent
        # so an old scan acknowledgement can never authorize a later turn.
        self.rear_goal_prerotate_gate_max_pending_steps = max(
            2,
            int(config.get("rear_goal_prerotate_gate_max_pending_steps", 32)),
        )
        self.rear_goal_prerotate_gate_pair_max_age_s = max(
            0.0,
            float(config.get("rear_goal_prerotate_gate_pair_max_age_s", 0.75)),
        )
        # A lost ROS relay delivery may consume one evaluator action as a
        # timeout noop.  Retry a small, explicit number of *delivery* windows
        # without allowing yaw-only control to become an unbounded loop.
        self.rear_goal_prerotate_delivery_retry_steps = max(
            0,
            int(config.get("rear_goal_prerotate_delivery_retry_steps", 2)),
        )
        self.rear_goal_lookahead_m = float(
            config.get("rear_goal_lookahead_m", 0.75)
        )
        self.rear_goal_pi_tie_tolerance_rad = float(
            config.get("rear_goal_pi_tie_tolerance_rad", 0.20)
        )
        self.rear_goal_pi_turn_sign = int(
            config.get("rear_goal_pi_turn_sign", -1)
        )
        # Direct base actions bypass DWA's trajectory checker.  They are
        # therefore admitted only against a recent local OccupancyGrid and a
        # conservative public circular footprint (the current costmap exposes
        # robot_radius, not a polygon footprint).
        self.rear_goal_local_costmap_max_age_s = max(
            0.0,
            float(config.get("rear_goal_local_costmap_max_age_s", 0.75)),
        )
        self.rear_goal_robot_radius_m = max(
            0.0,
            float(config.get("rear_goal_robot_radius_m", 0.25)),
        )
        self.rear_goal_safety_margin_m = max(
            0.0,
            float(config.get("rear_goal_safety_margin_m", 0.05)),
        )
        self.rear_goal_costmap_occupied_threshold = int(
            config.get("rear_goal_costmap_occupied_threshold", 253)
        )
        self.rear_goal_unknown_is_blocked = bool(
            config.get("rear_goal_unknown_is_blocked", True)
        )
        self.rear_goal_rotation_sweep_step_rad = max(
            1e-3,
            float(config.get("rear_goal_rotation_sweep_step_rad", 0.10)),
        )
        self.rear_goal_cmd_vel_cancel_wait_s = max(
            0.0,
            float(config.get("rear_goal_cmd_vel_cancel_wait_s", 0.50)),
        )
        self.rear_goal_oscillation_window_steps = max(
            2,
            int(config.get("rear_goal_oscillation_window_steps", 4)),
        )
        self.rear_goal_oscillation_min_sign_flips = max(
            1,
            int(config.get("rear_goal_oscillation_min_sign_flips", 2)),
        )
        self.rear_goal_oscillation_max_displacement_m = max(
            0.0,
            float(config.get("rear_goal_oscillation_max_displacement_m", 0.05)),
        )
        self.rear_goal_oscillation_min_goal_reduction_m = max(
            0.0,
            float(config.get("rear_goal_oscillation_min_goal_reduction_m", 0.02)),
        )
        self.rear_goal_cmd_vel_max_age_s = max(
            0.0,
            float(config.get("rear_goal_cmd_vel_max_age_s", 0.50)),
        )
        self.rear_goal_reverse_enabled = bool(
            config.get("rear_goal_reverse_enabled", True)
        )
        self.rear_goal_reverse_distance_m = max(
            0.0,
            float(config.get("rear_goal_reverse_distance_m", 0.15)),
        )
        self.rear_goal_reverse_min_distance_m = max(
            0.0,
            float(config.get("rear_goal_reverse_min_distance_m", 0.05)),
        )
        self.rear_goal_reverse_speed_mps = max(
            1e-3,
            float(config.get("rear_goal_reverse_speed_mps", 0.10)),
        )
        self.rear_goal_reverse_max_control_steps = max(
            1,
            int(config.get("rear_goal_reverse_max_control_steps", 10)),
        )
        self.final_align_enabled = bool(config.get("final_align_enabled", True))
        self.final_align_max_distance_m = float(
            config.get("final_align_max_distance_m", 0.12)
        )
        self.final_align_yaw_tolerance_rad = float(
            config.get("final_align_yaw_tolerance_rad", 0.15)
        )
        self.final_align_rotate_speed_rad_s = float(
            config.get("final_align_rotate_speed_rad_s", 0.30)
        )
        self.final_align_trigger_delay_s = float(
            config.get("final_align_trigger_delay_s", 2.0)
        )
        self.final_align_timeout_s = float(
            config.get("final_align_timeout_s", 15.0)
        )
        # Interaction approaches have a different terminal geometry contract:
        # their safe standoff and desired facing direction are part of the
        # public bridge precondition.  Keep this independently configurable so
        # enabling it cannot re-enable final yaw control for ordinary NAVIGATE
        # or EXPLORE behaviours.
        self.interaction_final_align_enabled = bool(
            config.get("interaction_final_align_enabled", True)
        )
        self.interaction_final_align_max_distance_m = max(
            0.05,
            float(config.get("interaction_final_align_max_distance_m", 0.35)),
        )
        self.interaction_final_align_yaw_tolerance_rad = max(
            0.05,
            float(config.get("interaction_final_align_yaw_tolerance_rad", 0.15)),
        )
        self.interaction_final_align_rotate_speed_rad_s = float(
            config.get("interaction_final_align_rotate_speed_rad_s", 0.30)
        )
        self.interaction_final_align_trigger_delay_s = max(
            0.0,
            float(config.get("interaction_final_align_trigger_delay_s", 2.0)),
        )
        self.interaction_final_align_timeout_s = max(
            0.1,
            float(config.get("interaction_final_align_timeout_s", 15.0)),
        )
        # Cmd_vel is admitted by the simulator only in an RGB/fresh-command
        # window.  Keep interaction final alignment on its own gate so an old
        # navigation/scan command can never be reused to rotate at a bridge
        # interaction pose.
        self.interaction_final_align_step_sync_enabled = bool(
            config.get("interaction_final_align_step_sync_enabled", True)
        )
        self.interaction_final_align_control_dt_s = max(
            1e-3,
            float(config.get("interaction_final_align_control_dt_s", 0.2)),
        )
        self.interaction_final_align_max_control_steps = max(
            1,
            # At 0.30 rad/s × 0.20 s, a raw full-pi turn spans 53 discrete
            # command windows.  Keep a small tracking margin while retaining
            # a strict finite interaction-only cap.
            int(config.get("interaction_final_align_max_control_steps", 56)),
        )
        # The bridge acknowledges a command window, not the exact base yaw
        # displacement.  Preserve a few finite tracking windows beyond the
        # ideal kinematic estimate so a near-interaction pose cannot fail only
        # because MuJoCo under-rotated one or two fixed-dt actions.
        self.interaction_final_align_control_step_margin_steps = max(
            0,
            int(config.get("interaction_final_align_control_step_margin_steps", 4)),
        )
        self.interaction_final_align_post_budget_settle_steps = max(
            0,
            int(
                config.get(
                    "interaction_final_align_post_budget_settle_steps", 3
                )
            ),
        )
        self.interaction_final_align_step_sync_stall_timeout_s = max(
            0.1,
            float(
                config.get(
                    "interaction_final_align_step_sync_stall_timeout_s", 2.0
                )
            ),
        )
        self.interaction_final_align_gate_max_pending_steps = max(
            2,
            int(config.get("interaction_final_align_gate_max_pending_steps", 32)),
        )
        self.interaction_final_align_gate_pair_max_age_s = max(
            0.0,
            float(config.get("interaction_final_align_gate_pair_max_age_s", 0.75)),
        )
        self.interaction_final_align_delivery_retry_steps = max(
            0,
            int(config.get("interaction_final_align_delivery_retry_steps", 2)),
        )
        self.evaluator_opaque_open_only = bool(
            config.get("evaluator_opaque_open_only", False)
        )
        self.make_plan_preflight_enabled = bool(
            config.get("make_plan_preflight_enabled", True)
        )
        self.make_plan_service = str(
            config.get("make_plan_service", "/move_base/make_plan")
        )
        self.make_plan_service_wait_sec = float(
            config.get("make_plan_service_wait_sec", 2.0)
        )
        self.make_plan_tolerance_m = float(
            config.get("make_plan_tolerance_m", 0.20)
        )
        self.make_plan_endpoint_tolerance_m = float(
            config.get("make_plan_endpoint_tolerance_m", 0.60)
        )
        self.make_plan_fail_open = bool(config.get("make_plan_fail_open", True))
        self.make_plan_empty_retry_count = max(
            0, int(config.get("make_plan_empty_retry_count", 2))
        )
        self.make_plan_empty_retry_delay_s = max(
            0.0, float(config.get("make_plan_empty_retry_delay_s", 0.15))
        )
        self.post_interaction_traversal_make_plan_retry_window_s = max(
            0.0,
            float(
                config.get(
                    "post_interaction_traversal_make_plan_retry_window_s", 0.35
                )
            ),
        )
        self.post_interaction_traversal_make_plan_retry_interval_s = max(
            0.01,
            float(
                config.get(
                    "post_interaction_traversal_make_plan_retry_interval_s", 0.10
                )
            ),
        )
        self.post_interaction_costmap_fresh_timeout_s = max(
            0.0,
            float(
                config.get(
                    "post_interaction_costmap_fresh_timeout_s",
                    2.0,
                )
            ),
        )
        self.post_interaction_costmap_fresh_poll_interval_s = max(
            0.01,
            float(
                config.get(
                    "post_interaction_costmap_fresh_poll_interval_s",
                    self.post_interaction_traversal_make_plan_retry_interval_s,
                )
            ),
        )
        self.explore_make_plan_fail_open_after_retries = bool(
            config.get("explore_make_plan_fail_open_after_retries", True)
        )
        self.explore_reservation_retry_sec = float(
            config.get("explore_reservation_retry_sec", 0.25)
        )
        # ExplorePy reports repeated make_plan failures here; the executor
        # remains the only owner allowed to cancel navigation and publish the
        # existing costmap-gated rear reverse command.
        self.external_recovery_request_enabled = bool(
            config.get("external_recovery_request_enabled", True)
        )
        self.external_recovery_request_max_age_s = max(
            0.0,
            float(config.get("external_recovery_request_max_age_s", 20.0)),
        )
        # A single active move_base goal can become ABORTED or make no progress
        # without producing three distinct failed subgoals. Run at most one
        # executor-owned, fresh-costmap-gated reverse/replan for that decision
        # before reporting the normal terminal failure.
        self.navigation_failure_recovery_enabled = bool(
            config.get("navigation_failure_recovery_enabled", True)
        )
        self.navigation_failure_recovery_max_attempts = max(
            0,
            int(config.get("navigation_failure_recovery_max_attempts", 1)),
        )
        self.final_align_cancel_wait_s = float(
            config.get("final_align_cancel_wait_s", 1.0)
        )
        self.stuck_recovery_enabled = bool(config.get("stuck_recovery_enabled", True))
        self.stuck_recovery_subgoal_failures = max(
            1, int(config.get("stuck_recovery_subgoal_failures", 3))
        )
        self.stuck_recovery_min_displacement_m = float(
            config.get("stuck_recovery_min_displacement_m", 0.10)
        )
        self.stuck_recovery_backoff_distance_m = float(
            config.get("stuck_recovery_backoff_distance_m", 0.20)
        )
        self.stuck_recovery_speed_mps = float(
            config.get("stuck_recovery_speed_mps", 0.12)
        )
        self.stuck_recovery_timeout_s = float(
            config.get("stuck_recovery_timeout_s", 8.0)
        )
        self.stuck_recovery_obstacle_escape_distance_m = float(
            config.get("stuck_recovery_obstacle_escape_distance_m", 0.35)
        )
        self.stuck_recovery_robot_radius_m = float(
            config.get("stuck_recovery_robot_radius_m", 0.25)
        )
        self.stuck_recovery_safety_margin_m = float(
            config.get("stuck_recovery_safety_margin_m", 0.05)
        )
        self.stuck_recovery_unknown_is_blocked = bool(
            config.get("stuck_recovery_unknown_is_blocked", True)
        )
        self.navigation_stagnation_timeout_s = float(
            config.get("navigation_stagnation_timeout_s", 12.0)
        )
        self.navigation_stagnation_distance_m = float(
            config.get("navigation_stagnation_distance_m", 0.10)
        )
        self.navigation_stagnation_yaw_rad = float(
            config.get("navigation_stagnation_yaw_rad", 0.15)
        )
        self.navigation_stagnation_goal_distance_reduction_m = max(
            0.0,
            float(config.get("navigation_stagnation_goal_distance_reduction_m", 0.02)),
        )
        self.navigation_stagnation_local_plan_max_age_s = max(
            0.0,
            float(config.get("navigation_stagnation_local_plan_max_age_s", 1.0)),
        )
        self.navigation_stagnation_local_plan_min_poses = max(
            1,
            int(config.get("navigation_stagnation_local_plan_min_poses", 2)),
        )
        self.interaction_approach_fallback_max_attempts = max(
            1,
            int(config.get("interaction_approach_fallback_max_attempts", 4)),
        )
        # A two-stage container candidate deliberately exposes a finite set of
        # outer observation poses (currently three rings x four faces).  Do not
        # silently truncate that set to the generic portal fallback budget: an
        # inner action-pose failure must still be allowed to earn a different
        # outer M1 observation.  The cap remains explicit and bounded.
        self.container_two_stage_fallback_max_attempts = max(
            1,
            int(config.get("container_two_stage_fallback_max_attempts", 12)),
        )
        self.interaction_approach_fallback_cancel_wait_s = max(
            0.0,
            float(config.get("interaction_approach_fallback_cancel_wait_s", 0.5)),
        )
        # The approach precondition is evaluated on fresh simulator-step pose
        # samples.  It has no wall-clock deadline: a bounded number of samples
        # determines when we abandon this approach option.
        self.interaction_approach_pose_poll_max_attempts = max(
            1,
            int(config.get("interaction_approach_pose_poll_max_attempts", 5)),
        )
        self.lock = threading.RLock()
        self.selection: dict | None = None
        self.latest_graph: dict = {}
        self._interaction_observation_requests: dict[str, dict] = {}
        self._container_m1_last_accepted_evidence: dict[str, dict] = {}
        self._latest_attribute_updates: dict[str, dict] = {}
        self._last_explore_reservation_publish_at = 0.0
        self._explore_reservation_publish_count = 0
        self._explore_feedback_received_count = 0
        self._explore_feedback_matched_count = 0
        self._explore_feedback_ignored_count = 0
        self._last_explore_feedback = {}
        self._external_recovery_requests: dict[str, dict] = {}
        self._external_recovery_consumed_ids: set[str] = set()
        self._external_recovery_received_count = 0
        self._external_recovery_accepted_count = 0
        self._external_recovery_rejected_count = 0
        self._external_recovery_consumed_count = 0
        self._external_recovery_last_feedback: dict = {}
        self._navigation_failure_recovery_attempts: dict[str, int] = {}
        self._stuck_failure_origin_xy: tuple[float, float] | None = None
        self._stuck_failure_candidate_ids: set[str] = set()
        self._latest_occupancy: OccupancyGrid | None = None
        self._latest_occupancy_received_at = 0.0
        self._latest_cmd_vel_stamped: dict | None = None
        self._rear_goal_turn_locks: dict[str, dict] = {}
        self._rear_dwa_monitor: dict | None = None
        self._executor_cmd_vel_lease_mode = ""
        self._last_rear_goal_recovery_detail: dict = {}
        # The post-open continuation is a causal map pipeline, not simply a
        # new global-costmap notification: raw SLAM OCC -> semantic planning
        # OCC (the StaticLayer input) -> global costmap.  Counters are local
        # because ROS header sequences can reset with move_base.
        self._raw_occupancy_received_count = 0
        self._latest_raw_occupancy_header_seq: int | None = None
        self._latest_raw_occupancy_header_stamp_sec: float | None = None
        self._latest_raw_occupancy_received_at = 0.0
        self._planning_occupancy_received_count = 0
        self._latest_planning_occupancy_header_seq: int | None = None
        self._latest_planning_occupancy_header_stamp_sec: float | None = None
        self._latest_planning_occupancy_received_at = 0.0
        self._raw_occupancy_events: deque[PostInteractionRawMapBarrier] = deque(
            maxlen=64
        )
        # ``costmap_updates`` is the low-latency final signal.  Keep the full
        # map too as a fallback, but only after a newer planning OCC has been
        # observed.
        self._global_costmap_received_count = 0
        self._latest_global_costmap_header_seq: int | None = None
        self._latest_global_costmap_received_at = 0.0
        self._global_costmap_update_received_count = 0
        self._latest_global_costmap_update_header_seq: int | None = None
        self._latest_global_costmap_update_received_at = 0.0
        self._post_interaction_costmap_baselines: dict[
            str, PostInteractionCostmapBaseline
        ] = {}
        self._post_interaction_raw_map_barriers: dict[
            str, tuple[PostInteractionRawMapBarrier, str]
        ] = {}
        self._post_interaction_planning_map_barriers: dict[
            str, PostInteractionPlanningMapBarrier
        ] = {}
        self._global_costmap_condition = threading.Condition(self.lock)
        self._latest_local_plan_received_at = 0.0
        self._latest_local_plan_pose_count = 0
        self._latest_step_sync_index: int | None = None
        self._latest_step_sync_received_at = 0.0
        self._startup_scan_gate = StepCommandGate(
            max_pending_steps=self.startup_scan_max_pending_steps,
            max_pair_age_s=self.startup_scan_pair_max_age_s,
        )
        self._rear_goal_prerotate_gate = StepCommandGate(
            max_pending_steps=self.rear_goal_prerotate_gate_max_pending_steps,
            max_pair_age_s=self.rear_goal_prerotate_gate_pair_max_age_s,
        )
        self._interaction_final_align_gate = StepCommandGate(
            max_pending_steps=self.interaction_final_align_gate_max_pending_steps,
            max_pair_age_s=self.interaction_final_align_gate_pair_max_age_s,
        )
        self._startup_scan_progress: dict = {}
        self._latest_rgb_step_seq: int | None = None
        self._latest_rgb_step_received_at = 0.0
        self.latest_image = None
        self.latest_image_sequence = 0
        self.pre_interaction_image_sequence = 0
        self._drawer_scan_wait_contexts: dict[str, dict] = {}
        self._drawer_scan_wait_records: dict[str, dict] = {}
        # Set only after a visual-contract-valid drawer scan command is
        # published.  It binds the finite bridge macro to simulator steps, not
        # wall-clock scheduler throughput.
        self._drawer_scan_execution_wait: dict[str, object] = {}
        self.active_skill_plan: dict = {}
        self.pending_skill_actions: list[dict] = []
        self.interaction_command_sequence = 0
        self.interaction_observation_sequence = 0
        self.verification_retries = 0
        self.model_events: list[dict] = []
        self._post_interaction_visual_audits: deque[dict] = deque(maxlen=128)
        self._post_interaction_visual_audit_reobserve_counts: dict[str, int] = {}
        self.feedback_pub = rospy.Publisher(
            topics.get("behavior_feedback", "/semantic_decision/behavior_feedback"),
            String,
            queue_size=10,
            latch=True,
        )
        self.external_recovery_feedback_pub = rospy.Publisher(
            topics.get("recovery_feedback", "/semantic_decision/recovery_feedback"),
            String,
            queue_size=8,
        )
        self.state_pub = rospy.Publisher(
            topics.get("execution_state", "/semantic_decision/execution_state"),
            String,
            queue_size=1,
            latch=True,
        )
        self.explore_command_pub = rospy.Publisher(
            topics.get("explore_command", "/explore_py/command"),
            String,
            queue_size=4,
            latch=True,
        )
        self.interaction_command_pub = rospy.Publisher(
            topics.get("interaction_command", "/semantic_decision/interaction_command"),
            String,
            queue_size=4,
            latch=True,
        )
        self.attribute_refresh_request_pub = rospy.Publisher(
            topics.get(
                "attribute_refresh_requests",
                "/semantic_mapping/attribute_refresh_requests",
            ),
            String,
            queue_size=8,
        )
        self.cmd_vel_pub = rospy.Publisher(
            topics.get("cmd_vel", "/cmd_vel"), Twist, queue_size=2
        )
        self.tf_listener = tf.TransformListener()
        self.move_base = actionlib.SimpleActionClient(
            topics.get("move_base", "/move_base"), MoveBaseAction
        )
        self.make_plan_client = rospy.ServiceProxy(self.make_plan_service, GetPlan)
        rospy.Subscriber(
            topics.get("selected_behavior", "/semantic_decision/selected_behavior"),
            String,
            self._selection_callback,
            queue_size=4,
        )
        rospy.Subscriber(
            topics.get("preempt_request", "/semantic_decision/preempt_request"),
            String,
            self._preempt_callback,
            queue_size=4,
        )
        rospy.Subscriber(
            topics.get("explore_feedback", "/explore_py/behavior_feedback"),
            String,
            self._explore_feedback_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            topics.get("recovery_request", "/semantic_decision/recovery_request"),
            String,
            self._external_recovery_request_callback,
            queue_size=8,
        )
        rospy.Subscriber(
            topics.get("interaction_result", "/semantic_mapping/interaction_result"),
            String,
            self._interaction_result_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            topics.get("attribute_updates", "/semantic_mapping/attribute_updates"),
            String,
            self._attribute_update_callback,
            queue_size=16,
        )
        rospy.Subscriber(
            topics.get("unified_graph", "/semantic_mapping/unified_graph"),
            String,
            self._graph_callback,
            queue_size=2,
        )
        rospy.Subscriber(
            topics.get("occupancy", "/move_base/local_costmap/costmap"),
            OccupancyGrid,
            self._occupancy_callback,
            queue_size=1,
        )
        # The bridge publishes the command actually eligible for the next
        # simulator action here.  We only use it while a move_base goal owns
        # navigation, to detect DWA left/right churn; executor-originated
        # recovery commands are gated separately and never enter that monitor.
        rospy.Subscriber(
            topics.get("cmd_vel_stamped", "/cmd_vel_stamped"),
            TwistStamped,
            self._cmd_vel_stamped_callback,
            queue_size=16,
        )
        rospy.Subscriber(
            topics.get("raw_occupancy_grid", "/struct_mapping/occ_map"),
            OccupancyGrid,
            self._raw_occupancy_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            topics.get(
                "planning_occupancy_grid",
                "/semantic_mapping/planning_occ_map",
            ),
            OccupancyGrid,
            self._planning_occupancy_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            topics.get("global_costmap", "/move_base/global_costmap/costmap"),
            OccupancyGrid,
            self._global_costmap_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            topics.get(
                "global_costmap_updates",
                "/move_base/global_costmap/costmap_updates",
            ),
            OccupancyGridUpdate,
            self._global_costmap_update_callback,
            queue_size=8,
        )
        rospy.Subscriber(
            topics.get("local_plan", "/move_base/DWAPlannerROS/local_plan"),
            Path,
            self._local_plan_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            topics.get("rgb_image", "/molmo_spaces/head_camera/image"),
            Image,
            self._image_callback,
            queue_size=1,
        )
        if (
            self.rear_goal_prerotate_step_sync_enabled
            or self.startup_scan_enabled
            or (
                self.interaction_final_align_enabled
                and self.interaction_final_align_step_sync_enabled
            )
            or self.drawer_scan_execution_step_budget_authoritative
            or self.interaction_approach_pose_poll_max_attempts > 1
        ):
            rospy.Subscriber(
                topics.get("step_sync", "/molmo_spaces/step_sync"),
                String,
                self._step_sync_callback,
                queue_size=32,
            )
        if (
            self.startup_scan_enabled
            or self.rear_goal_prerotate_step_sync_enabled
            or (
                self.interaction_final_align_enabled
                and self.interaction_final_align_step_sync_enabled
            )
        ):
            rospy.Subscriber(
                topics.get("fresh_command_gate", "/molmo_spaces/fresh_cmd_gate"),
                String,
                self._fresh_command_gate_callback,
                queue_size=32,
            )
        self.timer = rospy.Timer(rospy.Duration(0.2), self._tick)

    def _selection_callback(self, message: String) -> None:
        try:
            selection = json.loads(message.data)
        except json.JSONDecodeError:
            return
        # The decision node publishes an explicit inactive selection at every
        # terminal boundary so recorders cannot retain a stale subgoal.  It is
        # a display/state reset, never a new executable behavior.
        if selection.get("active") is False or not str(
            selection.get("candidate_id") or ""
        ):
            return
        with self.lock:
            if self.machine.state != STATE_IDLE or self.selection is not None:
                self._publish_feedback(selection, "REJECTED", False, {"reason": "executor_busy"})
                return
            self.selection = selection
            self.active_skill_plan = {}
            self.pending_skill_actions = []
            self._clear_drawer_scan_execution_wait_locked()
            self.interaction_command_sequence = 0
            self.interaction_observation_sequence = 0
            self.verification_retries = 0
            self.model_events = []
            decision_id = str(selection.get("decision_id") or "")
            if decision_id:
                self._navigation_failure_recovery_attempts.pop(decision_id, None)
                self._drawer_scan_wait_contexts.pop(decision_id, None)
                self._drawer_scan_wait_records.pop(decision_id, None)
                self._interaction_observation_requests.pop(decision_id, None)
                self._container_m1_last_accepted_evidence.pop(decision_id, None)
            commands = self.machine.start(selection)
            if self.machine.state == STATE_VERIFYING and requires_graph_verification(
                self.ablation.module3, selection
            ):
                commands.extend(self._verify_graph_locked())
            self._last_explore_reservation_publish_at = 0.0
            self._explore_reservation_publish_count = 0
            self._publish_feedback(selection, "STARTED", None, {})
        self._dispatch(commands)

    def _preempt_callback(self, message: String) -> None:
        try:
            request = json.loads(message.data)
        except json.JSONDecodeError:
            return
        selection = None
        cancel_navigation = False
        finalize_explore = False
        with self.lock:
            if self.selection is None:
                return
            requested_decision_id = str(request.get("decision_id") or "")
            active_decision_id = str(self.selection.get("decision_id") or "")
            if requested_decision_id != active_decision_id:
                return
            if str(request.get("reason") or "") != "preempted_by_target":
                return
            if str(self.selection.get("behavior_type") or "").upper() != "EXPLORE":
                return
            selection = dict(self.selection)
            cancel_navigation = self.machine.state in {
                STATE_NAVIGATING,
                STATE_APPROACH_INTERACTION,
            }
            finalize_explore = True
            self.selection = None
            self.machine.reset()
        if cancel_navigation:
            self.move_base.cancel_goal()
        detail = {
            "reason": "preempted_by_target",
            "replacement_candidate_id": str(
                request.get("replacement_candidate_id") or ""
            ),
        }
        if finalize_explore and selection is not None:
            self._publish_explore_command(
                selection,
                action="finalize_frontier",
                success=False,
                detail=detail,
            )
        if selection is not None:
            self._publish_feedback(selection, "CANCELED", False, detail)

    def _explore_feedback_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        commands = []
        with self.lock:
            self._explore_feedback_received_count += 1
            self._last_explore_feedback = dict(payload)
            if not self._matches_active(payload):
                self._explore_feedback_ignored_count += 1
                rospy.logwarn("[semantic_behavior_executor] ignored explore feedback: active=%s command=%s candidate=%s", self._command_id(self.selection) if self.selection else "", payload.get("command_id", ""), payload.get("candidate_id", ""))
                return
            self._explore_feedback_matched_count += 1
            status = str(payload.get("status") or "")
            rospy.loginfo("[semantic_behavior_executor] matched explore feedback: command=%s status=%s", payload.get("command_id", ""), status)
            if status == "READY":
                self._last_explore_reservation_publish_at = 0.0
                commands = self.machine.on_explore_ready(
                    detail=payload.get("detail") or {}
                )
            elif status in {"SUCCEEDED", "FAILED", "CANCELED", "REJECTED"}:
                commands = self.machine.on_explore_result(
                    status == "SUCCEEDED", detail=payload
                )
            else:
                return
        self._dispatch(commands)

    def _external_recovery_request_callback(self, message: String) -> None:
        """Accept only a request bound to the current EXPLORE reservation.

        The ROS callback records intent and returns immediately.  It never
        drives the base; `_run_navigation` consumes the request from the
        executor-owned navigation thread, where the normal fresh-costmap and
        single cmd_vel lease checks are available.
        """

        try:
            raw_payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        request = normalize_recovery_request(raw_payload)
        if request is None:
            return
        request_id = str(request["request_id"])
        accepted = False
        status = "REJECTED"
        detail: dict = {}
        with self.lock:
            self._external_recovery_received_count += 1
            if not self.external_recovery_request_enabled:
                self._external_recovery_rejected_count += 1
                detail = {"reason": "external_recovery_disabled"}
            elif not recovery_request_matches_selection(request, self.selection):
                self._external_recovery_rejected_count += 1
                detail = {"reason": "recovery_request_not_active_explore"}
            elif self.machine.state not in {STATE_PREPARING_EXPLORE, STATE_NAVIGATING}:
                self._external_recovery_rejected_count += 1
                detail = {"reason": "recovery_request_wrong_executor_state"}
            elif request_id in self._external_recovery_consumed_ids:
                accepted = True
                status = "DUPLICATE"
                detail = {"reason": "recovery_request_already_consumed"}
            elif request_id in self._external_recovery_requests:
                accepted = True
                status = "DUPLICATE"
                detail = {"reason": "recovery_request_already_pending"}
            else:
                request["received_at_monotonic"] = time.monotonic()
                self._external_recovery_requests[request_id] = request
                self._external_recovery_accepted_count += 1
                accepted = True
                status = "ACCEPTED"
                detail = {"reason": "queued_for_executor_navigation"}
        self._publish_external_recovery_feedback(
            request,
            status=status,
            accepted=accepted,
            detail=detail,
        )

    def _publish_external_recovery_feedback(
        self,
        request: dict,
        *,
        status: str,
        accepted: bool | None,
        detail: dict | None = None,
    ) -> None:
        payload = {
            "request_id": request.get("request_id", ""),
            "source": "semantic_behavior_executor",
            "decision_id": request.get("decision_id", ""),
            "candidate_id": request.get("candidate_id", ""),
            "recovery": request.get("recovery", ""),
            "status": str(status),
            "accepted": accepted,
            "detail": dict(detail or {}),
            "timestamp": time.time(),
        }
        with self.lock:
            self._external_recovery_last_feedback = dict(payload)
        self.external_recovery_feedback_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )

    def _take_external_recovery_request(
        self, decision_id: str, candidate: dict
    ) -> dict | None:
        """Claim one current request; a request may drive at most one reverse."""

        now = time.monotonic()
        with self.lock:
            stale_ids = [
                request_id
                for request_id, request in self._external_recovery_requests.items()
                if (
                    self.external_recovery_request_max_age_s > 0.0
                    and now - float(request.get("received_at_monotonic", now))
                    > self.external_recovery_request_max_age_s
                )
            ]
            for request_id in stale_ids:
                self._external_recovery_requests.pop(request_id, None)
            selection = self.selection
            for request_id, request in list(self._external_recovery_requests.items()):
                if (
                    str(request.get("decision_id") or "") != str(decision_id)
                    or not recovery_request_matches_selection(request, selection)
                    or str(candidate.get("candidate_id") or "")
                    != str(request.get("candidate_id") or "")
                ):
                    continue
                self._external_recovery_requests.pop(request_id, None)
                self._external_recovery_consumed_ids.add(request_id)
                # Bounded bookkeeping: only the latest request identities are
                # needed to make duplicate ROS deliveries idempotent.
                if len(self._external_recovery_consumed_ids) > 128:
                    self._external_recovery_consumed_ids = set(
                        list(self._external_recovery_consumed_ids)[-64:]
                    )
                self._external_recovery_consumed_count += 1
                return dict(request)
        return None

    def _consume_external_recovery_if_needed(
        self,
        decision_id: str,
        candidate: dict,
        *,
        preflight_failed: bool,
        start_goal_option_index: int,
        interaction_approach_attempts: list[dict],
    ) -> bool:
        """Run at most one queued safe reverse, solely from the executor.

        Return ``True`` when this navigation worker has handed off or finished
        the request.  A currently reachable path simply resolves a stale
        explorer report without moving the base.
        """

        request = self._take_external_recovery_request(decision_id, candidate)
        if request is None:
            return False
        if not preflight_failed:
            self._publish_external_recovery_feedback(
                request,
                status="NOT_NEEDED",
                accepted=True,
                detail={"reason": "executor_make_plan_reachable"},
            )
            return False
        self._publish_external_recovery_feedback(
            request,
            status="EXECUTING",
            accepted=True,
            detail={"reason": "executor_owned_safe_reverse_replan"},
        )
        recovered, recovery_detail = self._attempt_rear_goal_reverse(
            decision_id,
            allow_idle_action_client=True,
        )
        recovery_detail = {
            "request_id": request.get("request_id", ""),
            "request_reason": request.get("reason", ""),
            **dict(recovery_detail or {}),
        }
        if recovered and self._navigation_is_current(decision_id):
            self._publish_external_recovery_feedback(
                request,
                status="SUCCEEDED",
                accepted=True,
                detail=recovery_detail,
            )
            threading.Thread(
                target=self._run_navigation,
                args=(
                    decision_id,
                    candidate,
                    int(start_goal_option_index),
                    list(interaction_approach_attempts),
                ),
                daemon=True,
            ).start()
            return True
        self._publish_external_recovery_feedback(
            request,
            status="FAILED",
            accepted=True,
            detail=recovery_detail,
        )
        self._handle_navigation_result(
            decision_id,
            False,
            {
                "reason": "external_recovery_failed",
                "external_recovery": True,
                # Avoid falling through to legacy ungated stuck recovery.
                "rear_goal_recovery": True,
                "external_recovery_detail": recovery_detail,
            },
        )
        return True

    @staticmethod
    def _finite_public_bbox(value: object) -> list[float] | None:
        """Normalize one detector box carried by a targeted M1 update."""

        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return None
        try:
            x0, y0, x1, y1 = (float(item) for item in value[:4])
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(item) for item in (x0, y0, x1, y1)):
            return None
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        if right - left < 1.0 or bottom - top < 1.0:
            return None
        return [left, top, right, bottom]

    @staticmethod
    def _is_drawer_pre_action_candidate(candidate: dict | None) -> bool:
        return bool(
            ((candidate or {}).get("metadata") or {}).get(
                "drawer_pre_action_observation", False
            )
        )

    @staticmethod
    def _is_container_pre_action_candidate(candidate: dict | None) -> bool:
        return bool(
            ((candidate or {}).get("metadata") or {}).get(
                "container_pre_action_observation", False
            )
        )

    @staticmethod
    def _container_visual_truncation_reason(update: dict) -> str:
        """Reject only missing lateral contact evidence for a container.

        A low camera can clip the bottom of a tall drawer/fridge box while the
        front plane and all model-provided action regions remain visible.  That
        is different from clipping the left/right boundary, which prevents a
        reliable frontality judgement.  Older messages without the public edge
        list remain fail-closed.
        """

        if not bool(update.get("visual_evidence_truncated", False)):
            return ""
        raw_edges = update.get("visual_evidence_truncated_edges")
        if not isinstance(raw_edges, (list, tuple, set)):
            return "m1_visual_evidence_truncated"
        edges = {str(edge).strip().casefold() for edge in raw_edges}
        if not edges:
            return "m1_visual_evidence_truncated"
        if edges.intersection({"left", "right", "unknown"}):
            return "m1_visual_evidence_laterally_truncated"
        return ""

    def _container_m1_pre_action_ready_locked(
        self,
        update: dict,
        request: dict,
    ) -> str:
        """Validate the public fresh-M1 contract before opening a container.

        The state machine makes the final transition.  This helper only turns
        malformed/stale updates into an ordinary bounded re-observation rather
        than allowing a ring pose or stale detector frame to reach the bridge.
        """

        status = str(update.get("attribute_status") or "").strip().casefold()
        if status != "ready":
            return f"m1_attribute_status_{status or 'missing'}"
        if update.get("is_currently_visible") is not True:
            return "m1_target_not_currently_visible"
        truncation_reason = self._container_visual_truncation_reason(update)
        if truncation_reason:
            return truncation_reason
        if self._finite_public_bbox(update.get("observed_bbox_2d")) is None:
            return "m1_public_bbox_unavailable"
        capture_step = self._public_step_or_none(
            update.get("observation_capture_step")
            or update.get("attribute_capture_step")
        )
        if capture_step is None:
            return "m1_capture_step_unavailable"
        minimum_capture_step = self._public_step_or_none(
            request.get("minimum_capture_step")
        )
        if (
            minimum_capture_step is not None
            and capture_step <= minimum_capture_step
        ):
            return "m1_capture_not_fresh"
        view_state = str(update.get("view_state") or "unknown").strip().casefold()
        allowed_views = (
            {"front"}
            if bool(getattr(self, "container_pre_action_require_direct_front", True))
            else {"front", "oblique"}
        )
        if view_state not in allowed_views:
            return f"m1_view_state_{view_state}"
        if update.get("front_surface_visible") is not True:
            return "m1_front_surface_not_visible"
        if update.get("approach_ready") is not True:
            return "m1_approach_not_ready"
        if bool(update.get("needs_reobserve", False)):
            return "m1_needs_reobserve"
        return "ready"

    def _container_m1_staging_pose_tolerance_m(self, candidate: dict) -> float:
        """Return the pose envelope for a model-only outer staging observation.

        The normal capture tolerance deliberately remains tight.  A candidate
        may opt into a slightly wider envelope only when it explicitly records
        an *outer* container staging ring.  That envelope is for the M1 image
        barrier, never a generic contact relaxation: the bridge later receives
        the paired inner action pose and performs its own authoritative check.
        """

        base_tolerance_m = max(
            0.05,
            float(getattr(self, "container_m1_capture_pose_tolerance_m", 0.18)),
        )
        metadata = candidate.get("metadata") or {}
        if not bool(metadata.get("m1_observation_staging_required", False)):
            return base_tolerance_m
        try:
            outer_offset_m = float(
                metadata.get("m1_safe_staging_outer_offset_m", 0.0) or 0.0
            )
            declared_tolerance_m = float(
                metadata.get("m1_safe_staging_arrival_tolerance_m", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            return base_tolerance_m
        if (
            outer_offset_m <= 1e-6
            or declared_tolerance_m <= base_tolerance_m
            # Do not use metadata to create an outer visual envelope wider than
            # the clearance deliberately added to that staging ring.  The
            # two-stage candidate increases both by the same amount, while the
            # inner bridge gate remains independently strict.
            or declared_tolerance_m > outer_offset_m + 1e-6
        ):
            return base_tolerance_m
        return declared_tolerance_m

    def _interaction_navigation_pose_tolerance_m(self, candidate: dict) -> float:
        """Use the wider envelope only while moving to an outer M1 staging pose."""

        metadata = candidate.get("metadata") or {}
        if bool(metadata.get("m1_observation_staging_required", False)):
            return self._container_m1_staging_pose_tolerance_m(candidate)
        interaction = candidate.get("interaction_command") or {}
        return max(
            0.05,
            float(interaction.get("interaction_ready_distance_m", 0.45) or 0.45),
        )

    def _container_safe_staging_arrival_sample(
        self,
        candidate: dict,
        expected_pose_xyyaw: tuple[float, float, float],
        detail: dict,
    ) -> dict | None:
        """Revalidate a just-observed outer staging pose for the M1 barrier.

        ``move_base`` can report success in the interval between two evaluator
        steps.  Re-polling TF immediately after cancelling that goal can then
        lose the same valid pose and consume every bounded retry before M1 is
        requested.  Reuse only a pose that the navigation loop has *already*
        validated against this exact selected outer staging goal.  This does
        not bypass M1 or the bridge: it merely avoids discarding causal pose
        evidence acquired in the same approach transition.
        """

        metadata = candidate.get("metadata") or {}
        if not bool(metadata.get("m1_observation_staging_required", False)):
            return None
        try:
            outer_offset_m = float(
                metadata.get("m1_safe_staging_outer_offset_m", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            return None
        if outer_offset_m <= 1e-6:
            return None
        arrival_validation = detail.get("interaction_pose_validation")
        if not isinstance(arrival_validation, dict):
            return None
        actual_pose = arrival_validation.get("actual_pose_xyyaw")
        interaction = candidate.get("interaction_command") or {}
        validation = interaction_pose_validation(
            list(expected_pose_xyyaw),
            None if actual_pose is None else list(actual_pose),
            distance_tolerance_m=self._container_m1_staging_pose_tolerance_m(
                candidate
            ),
            yaw_tolerance_rad=float(
                interaction.get("interaction_ready_yaw_tolerance_rad", 0.55)
                or 0.55
            ),
        )
        if not bool(validation.get("valid")):
            return None
        validation["poll_index"] = 0
        validation["step_index"] = detail.get("interaction_arrival_step_index")
        validation["sample_source"] = "navigation_arrival_pose"
        return validation

    def _container_m1_capture_evidence_locked(
        self,
        candidate: dict,
        update: dict,
        request: dict,
    ) -> tuple[dict | None, str]:
        """Bind accepted public M1 evidence to its safe staging pose.

        Targeted attribute updates contain the image capture step but not the
        robot pose.  Capture the executor's current public TF pose while that
        request is active, and require it to remain close to the exact staging
        pose selected by navigation.  This prevents a later close-range pose or
        a different ring face from reusing a visually good response.
        """

        metadata = candidate.get("metadata") or {}
        interaction = candidate.get("interaction_command") or {}
        if is_container_two_stage_physical_action(candidate):
            # A two-stage candidate has already left its outer M1 stance.  Do
            # not let a delayed targeted response become evidence for the
            # nearer physical action pose.
            return None, "m1_capture_not_outer_staging_phase"
        expected = list(
            request.get("observation_pose_xyyaw")
            or metadata.get("effective_interaction_approach_pose_xyyaw")
            or interaction.get("interaction_approach_pose_xyyaw")
            or []
        )
        if len(expected) < 3:
            return None, "m1_capture_pose_unavailable"
        frame_id = str(metadata.get("frame_id") or getattr(self, "map_frame", "map"))
        # Reduced unit-test executors deliberately omit TF.  In production a
        # listener is always present; without one, use the selected staging
        # pose as the only deterministic mock evidence rather than calling the
        # bound helper and tripping over an incomplete ``tf`` stub.
        pose_reader = (
            getattr(self, "_current_pose", None)
            if getattr(self, "tf_listener", None) is not None
            else None
        )
        # The production executor always has a TF reader.  Keep reduced unit
        # test doubles deterministic by treating their selected staging pose
        # as the capture pose rather than failing solely because they omit TF.
        actual = pose_reader(frame_id) if callable(pose_reader) else list(expected)
        validation = interaction_pose_validation(
            expected,
            None if actual is None else list(actual),
            distance_tolerance_m=self._container_m1_staging_pose_tolerance_m(
                candidate
            ),
            yaw_tolerance_rad=getattr(
                self, "container_m1_capture_yaw_tolerance_rad", 0.25
            ),
        )
        if not bool(validation.get("valid")):
            return None, "m1_capture_pose_mismatch"
        capture_step = self._public_step_or_none(
            update.get("observation_capture_step")
            or update.get("attribute_capture_step")
        )
        if capture_step is None:
            return None, "m1_capture_step_unavailable"
        return {
            "capture_step": int(capture_step),
            "capture_pose_xyyaw": list(validation.get("actual_pose_xyyaw") or []),
            "staging_pose_xyyaw": list(validation.get("expected_pose_xyyaw") or []),
            "pose_validation": validation,
            "view_state": str(update.get("view_state") or ""),
            "front_surface_visible": bool(update.get("front_surface_visible")),
            "approach_ready": bool(update.get("approach_ready")),
        }, "ready"

    def _container_m1_evidence_still_at_capture_pose_locked(
        self, candidate: dict, evidence: dict
    ) -> bool:
        """Return whether a temporal M1 flip is still at the accepted pose."""

        capture_pose = list(evidence.get("capture_pose_xyyaw") or [])
        if len(capture_pose) < 3:
            return False
        metadata = candidate.get("metadata") or {}
        pose_reader = (
            getattr(self, "_current_pose", None)
            if getattr(self, "tf_listener", None) is not None
            else None
        )
        actual = (
            pose_reader(str(metadata.get("frame_id") or getattr(self, "map_frame", "map")))
            if callable(pose_reader)
            else list(capture_pose)
        )
        validation = interaction_pose_validation(
            capture_pose,
            None if actual is None else list(actual),
            distance_tolerance_m=self._container_m1_staging_pose_tolerance_m(
                candidate
            ),
            yaw_tolerance_rad=getattr(
                self, "container_m1_capture_yaw_tolerance_rad", 0.25
            ),
        )
        return bool(validation.get("valid"))

    @staticmethod
    def _candidate_with_container_m1_evidence(
        candidate: dict, evidence: dict
    ) -> dict:
        """Preserve one accepted M1 record without altering candidate identity."""

        result = dict(candidate)
        metadata = dict(result.get("metadata") or {})
        metadata["accepted_container_m1_evidence"] = dict(evidence)
        metadata["accepted_container_m1_capture_pose_xyyaw"] = list(
            evidence.get("capture_pose_xyyaw") or []
        )
        metadata["accepted_container_m1_capture_step"] = evidence.get("capture_step")
        result["metadata"] = metadata
        return result

    def _drawer_candidate_from_m1_update_locked(
        self,
        candidate: dict,
        update: dict,
        request: dict,
    ) -> tuple[dict | None, str]:
        """Ground a drawer scan in one fresh M1 observation or reject it.

        The box is an output-side public detector record paired with the M1
        image.  The model supplies only crop-relative visible action regions;
        no graph pose, joint, or hidden drawer information can enter here.
        """

        status = str(update.get("attribute_status") or "").strip().casefold()
        if status != "ready":
            return None, f"m1_attribute_status_{status or 'missing'}"
        if update.get("is_currently_visible") is not True:
            return None, "m1_target_not_currently_visible"
        truncation_reason = self._container_visual_truncation_reason(update)
        if truncation_reason:
            return None, truncation_reason
        view_state = str(update.get("view_state") or "unknown").strip().casefold()
        allowed_views = (
            {"front"}
            if bool(getattr(self, "container_pre_action_require_direct_front", True))
            else {"front", "oblique"}
        )
        if view_state not in allowed_views:
            return None, f"m1_view_state_{view_state}"
        if update.get("front_surface_visible") is not True:
            return None, "m1_front_surface_not_visible"
        if update.get("approach_ready") is not True:
            return None, "m1_approach_not_ready"
        if bool(update.get("needs_reobserve", False)):
            return None, "m1_needs_reobserve"
        bbox = self._finite_public_bbox(update.get("observed_bbox_2d"))
        if bbox is None:
            return None, "m1_public_bbox_unavailable"
        capture_step = self._public_step_or_none(
            update.get("observation_capture_step")
            or update.get("attribute_capture_step")
        )
        if capture_step is None:
            return None, "m1_capture_step_unavailable"
        minimum_capture_step = self._public_step_or_none(
            request.get("minimum_capture_step")
        )
        if (
            minimum_capture_step is not None
            and capture_step <= minimum_capture_step
        ):
            return None, "m1_capture_not_fresh"
        visual_plan = {
            "target_type": "drawer_container",
            # A visible drawer set is an inspection macro, not a persistent
            # multi-drawer opening.  The force bridge keeps the low view while
            # it opens, observes, and closes one M1-grounded drawer before the
            # next.  This prevents two drawers from remaining open together
            # and gives the interaction a real bounded simulator-step cost.
            "action": "scan",
            "operation_method": "pull",
            "drawer_sequence_type": "drawer_scan",
            "view_state": view_state,
            "approach_ready": True,
            "reposition_required": False,
            "confidence": float(update.get("confidence", 0.0) or 0.0),
            "reason": "fresh_m1_drawer_action_regions",
            "source": str(update.get("source") or "mllm_attribute_inference"),
            "capture_step": capture_step,
        }
        planned = candidate_with_visual_drawer_scan(
            candidate,
            drawer_bbox_2d=bbox,
            capture_step=capture_step,
            action_regions=update.get("action_regions"),
            visual_plan=visual_plan,
        )
        if planned is None:
            return None, "m1_visible_drawer_action_regions_missing"
        return planned, "ready"

    def _attribute_update_callback(self, message: String) -> None:
        """Consume only the exact fresh M1 response requested by this executor.

        Attribute discovery normally updates the graph asynchronously and must
        never advance an active interaction.  This path is different: a
        portal re-observation or drawer pre-action gate explicitly asked M1 for
        a post-pose RGB+detection pair.  Match the opaque request ID and
        preserve failed responses as a bounded re-observation attempt.
        """

        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        updates = payload.get("updates") if isinstance(payload, dict) else None
        if not isinstance(updates, list):
            return
        commands: list[dict] = []
        with self.lock:
            if (
                self.selection is None
                or self.machine.state != STATE_WAITING_FOR_INTERACTION_OBSERVATION
            ):
                return
            decision_id = str(self.selection.get("decision_id") or "")
            request = dict(self._interaction_observation_requests.get(decision_id) or {})
            if not request:
                return
            active_episode = str(self.selection.get("episode_id") or "")
            payload_episode = str(payload.get("episode_id") or "")
            if active_episode and payload_episode and active_episode != payload_episode:
                return
            for raw_update in updates:
                if not isinstance(raw_update, dict):
                    continue
                update = dict(raw_update)
                object_id = str(update.get("object_id") or "")
                if object_id:
                    self._latest_attribute_updates[object_id] = update
                if not bool(update.get("targeted_refresh", False)):
                    continue
                if str(update.get("targeted_refresh_request_id") or "") != str(
                    request.get("request_id") or ""
                ):
                    continue
                expected_object_ids = {
                    str(request.get("object_id") or ""),
                    str(request.get("node_id") or ""),
                }
                expected_object_ids.discard("")
                if expected_object_ids and object_id not in expected_object_ids:
                    continue
                # pending is an acknowledgement that the fresh pair was
                # queued, not a new visual judgement. Wait for ready/failed.
                status = str(update.get("attribute_status") or "").casefold()
                if status == "pending":
                    return
                update.setdefault("attribute_source", update.get("source") or "")
                update.setdefault("attribute_capture_step", update.get("observation_capture_step"))
                if status == "ready":
                    # The targeted request itself can only be consumed by a
                    # later visible detection; do not require a hidden GT flag.
                    update.setdefault("is_currently_visible", True)
                candidate = dict(self.machine.candidate or self.selection or {})
                if self._is_drawer_pre_action_candidate(candidate):
                    planned, drawer_reason = self._drawer_candidate_from_m1_update_locked(
                        candidate,
                        update,
                        request,
                    )
                    if planned is not None:
                        evidence, capture_reason = (
                            self._container_m1_capture_evidence_locked(
                                planned, update, request
                            )
                        )
                        if evidence is None:
                            planned = None
                            drawer_reason = capture_reason
                        else:
                            planned = self._candidate_with_container_m1_evidence(
                                planned, evidence
                            )
                            accepted_by_decision = getattr(
                                self, "_container_m1_last_accepted_evidence", None
                            )
                            if accepted_by_decision is None:
                                accepted_by_decision = {}
                                self._container_m1_last_accepted_evidence = (
                                    accepted_by_decision
                                )
                            accepted_by_decision[decision_id] = dict(evidence)
                    update["drawer_action_regions_ready"] = planned is not None
                    update["drawer_visual_precondition_reason"] = drawer_reason
                    if planned is not None:
                        self.machine.candidate = planned
                        # The selection is the recorder/feedback authority;
                        # retain the exact fresh box+region command that will
                        # be sent if the state machine accepts this M1 view.
                        self.selection = dict(planned)
                        interaction = dict(
                            planned.get("interaction_command") or {}
                        )
                        self.active_skill_plan = {
                            "visual_operation_plan": dict(
                                interaction.get("visual_operation_plan") or {}
                            ),
                            "subactions": [],
                            "max_retries": 0,
                        }
                        self.pending_skill_actions = []
                elif self._is_container_pre_action_candidate(candidate):
                    container_reason = self._container_m1_pre_action_ready_locked(
                        update, request
                    )
                    evidence = None
                    if container_reason == "ready":
                        evidence, capture_reason = (
                            self._container_m1_capture_evidence_locked(
                                candidate, update, request
                            )
                        )
                        if evidence is None:
                            container_reason = capture_reason
                        else:
                            candidate = self._candidate_with_container_m1_evidence(
                                candidate, evidence
                            )
                            self.machine.candidate = candidate
                            self.selection = dict(candidate)
                            accepted_by_decision = getattr(
                                self, "_container_m1_last_accepted_evidence", None
                            )
                            if accepted_by_decision is None:
                                accepted_by_decision = {}
                                self._container_m1_last_accepted_evidence = (
                                    accepted_by_decision
                                )
                            accepted_by_decision[decision_id] = dict(evidence)
                    update["container_visual_precondition_reason"] = container_reason
                    required_confirmations = max(
                        1,
                        int(
                            getattr(
                                self,
                                "container_pre_action_confirmation_count",
                                1,
                            )
                            or 1
                        ),
                    )
                    confirmation_count = max(
                        0, int(request.get("confirmation_count", 0) or 0)
                    )
                    capture_step = self._public_step_or_none(
                        update.get("observation_capture_step")
                        or update.get("attribute_capture_step")
                    )
                    if (
                        container_reason == "ready"
                        and capture_step is not None
                        and confirmation_count + 1 < required_confirmations
                    ):
                        # Keep the robot at this observation pose and ask for a
                        # causally later RGB+detection pair.  Do not let a
                        # transient one-frame "front" judgement open a
                        # container from a side view.
                        update["container_visual_precondition_reason"] = (
                            "m1_front_confirmation_pending"
                        )
                        commands = [
                            {
                                "kind": "request_interaction_observation",
                                "candidate": candidate,
                                "node_id": str(request.get("node_id") or ""),
                                "object_id": str(request.get("object_id") or ""),
                                "attempt": int(request.get("attempt", 0) or 0),
                                "min_capture_step": int(capture_step),
                                "reason": "mllm_container_front_confirmation",
                                "confirmation_count": confirmation_count + 1,
                                "accepted_evidence": dict(evidence or {}),
                                "same_pose_flip_count": 0,
                            }
                        ]
                        self._interaction_observation_requests.pop(decision_id, None)
                        break
                    accepted_evidence = dict(
                        request.get("accepted_evidence")
                        or getattr(
                            self, "_container_m1_last_accepted_evidence", {}
                        ).get(decision_id)
                        or {}
                    )
                    same_pose_flip_count = max(
                        0, int(request.get("same_pose_flip_count", 0) or 0)
                    )
                    if (
                        container_reason != "ready"
                        and accepted_evidence
                        and same_pose_flip_count
                        < int(
                            getattr(
                                self,
                                "container_m1_same_pose_flip_retry_count",
                                1,
                            )
                            or 0
                        )
                        and self._container_m1_evidence_still_at_capture_pose_locked(
                            candidate, accepted_evidence
                        )
                    ):
                        # Do not discard a valid front merely because the next
                        # M1 response flickered at the same pose.  Ask once more
                        # at that *same* staging pose; moving to another ring
                        # option remains the bounded fallback if it persists.
                        update["container_visual_precondition_reason"] = (
                            "m1_temporal_flip_reobserve_same_staging_pose"
                        )
                        commands = [
                            {
                                "kind": "request_interaction_observation",
                                "candidate": candidate,
                                "node_id": str(request.get("node_id") or ""),
                                "object_id": str(request.get("object_id") or ""),
                                "attempt": int(request.get("attempt", 0) or 0),
                                "min_capture_step": int(
                                    capture_step
                                    if capture_step is not None
                                    else request.get("minimum_capture_step", 0)
                                    or 0
                                ),
                                "reason": "mllm_container_temporal_flip_reobserve",
                                "confirmation_count": confirmation_count,
                                "accepted_evidence": accepted_evidence,
                                "same_pose_flip_count": same_pose_flip_count + 1,
                            }
                        ]
                        self._interaction_observation_requests.pop(decision_id, None)
                        break
                commands = self.machine.on_interaction_observation_result(update)
                self._interaction_observation_requests.pop(decision_id, None)
                break
        self._dispatch(commands)

    def _interaction_result_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        # M3 is launched as a post-action audit.  Keep a complete immutable
        # snapshot here because the backend result may finish the state machine
        # (and clear ``self.selection``) before the model response arrives.
        audit_task = None
        with self.lock:
            if not self._matches_active(payload):
                return
            if str(payload.get("command_id") or "") == str(
                (getattr(self, "_drawer_scan_execution_wait", {}) or {}).get(
                    "command_id", ""
                )
            ):
                self._clear_drawer_scan_execution_wait_locked()
            if is_interaction_pose_precondition_failure(payload):
                # The force bridge rejected this before it touched the object.
                # Re-enter approach navigation; do not emit a terminal
                # interaction failure or teach the decision node that the
                # object is non-interactable.
                next_candidate = None
                commands = self._retry_interaction_approach_after_pose_failure_locked(
                    payload
                )
            else:
                static_portal_feedback = is_static_portal_interaction_feedback(
                    self.selection,
                    payload,
                )
                self._record_post_interaction_costmap_baseline_locked(payload)
                visual_plan = dict(self.active_skill_plan.get("visual_operation_plan") or {})
                selected_interaction = dict(
                    (self.selection or {}).get("interaction_command") or {}
                )
                drawer_sequence_type = str(
                    selected_interaction.get("sequence_type") or ""
                ).casefold()
                is_drawer_scan = drawer_sequence_type == "drawer_scan"
                is_drawer_interaction = (
                    str(visual_plan.get("target_type") or "") == "drawer_container"
                    or drawer_sequence_type in {"drawer_scan", "drawer_open"}
                )
                drawer_failure_reason = str(
                    payload.get("failure_reason") or payload.get("reason") or ""
                ).strip().casefold()
                if (
                    is_drawer_interaction
                    and not bool(payload.get("success"))
                    and drawer_failure_reason
                    in {
                        "drawer_scan_execution_failed",
                        "drawer_open_execution_failed",
                        "articulation_resolution_failed",
                    }
                ):
                    # The M1-gated command had a fresh visible front/region;
                    # an executor rejection at this point is a concrete
                    # candidate-level failure, not a stale visual observation
                    # to send back through M3 or a generic cooldown.
                    payload = {
                        **payload,
                        "failure_stage": "interaction_execution",
                        "terminal_candidate_exclusion": True,
                        "reason": "drawer_interaction_execution_unavailable",
                    }
                if static_portal_feedback:
                    # A static opening already returned its terminal public
                    # postcondition.  It must not be treated as a successful
                    # articulated skill step and advance an MLLM subaction plan.
                    self.pending_skill_actions = []
                if (
                    self.ablation.module3 == "mllm_skill_verified"
                    and bool(payload.get("success"))
                    and self.pending_skill_actions
                    and not static_portal_feedback
                ):
                    next_action = self.pending_skill_actions.pop(0)
                    next_candidate = self._candidate_for_skill_action(
                        dict(self.selection or {}), next_action
                    )
                    commands = []
                else:
                    next_candidate = None
                    backend_success = bool(
                        payload.get("success")
                        or str(payload.get("status") or "").upper() == "SUCCEEDED"
                    )
                    commands = self.machine.on_interaction_result(
                        backend_success, detail=payload
                    )
                if (
                    self.machine.state == STATE_VERIFYING
                    and self.ablation.module3 == "direct_atomic"
                ):
                    commands.extend(
                        self.machine.on_verification_result(
                            backend_success,
                            detail={**payload, "verification_mode": "trusted_backend_result"},
                        )
                    )
                elif (
                    self.machine.state == STATE_VERIFYING
                    and self.ablation.module3 == "rule_verified"
                ):
                    commands.extend(self._verify_graph_locked())
                elif (
                    self.machine.state == STATE_VERIFYING
                    and self.ablation.module3 == "mllm_skill_verified"
                ):
                    if is_drawer_scan:
                        # A drawer scan intentionally closes each drawer after its
                        # low-view observation.  Re-checking the final exterior
                        # crop as though it should remain open would reject a
                        # successful scan.  The semantic map receives the frames
                        # captured while each drawer is open instead.
                        commands.extend(
                            self.machine.on_backend_result(
                                backend_success,
                                detail={
                                    **payload,
                                    "verification_mode": "drawer_scan_backend",
                                },
                            )
                        )
                    else:
                        # A successful/failed sealed backend result is already
                        # the execution fact.  Finish immediately so a slow or
                        # mistaken M3 response cannot replay this command.  The
                        # visual request below is deliberately detached from the
                        # state machine and can only record/re-observe.
                        decision_id = str((self.selection or {}).get("decision_id") or "")
                        audit_selection = dict(self.selection or {})
                        audit_node = self._selected_graph_node_locked(audit_selection)
                        audit_task = (
                            decision_id,
                            dict(payload),
                            int(self.latest_image_sequence),
                            audit_selection,
                            audit_node,
                        )
                        commands.extend(
                            self.machine.on_backend_result(
                                backend_success,
                                detail={
                                    **payload,
                                    "backend_success": backend_success,
                                    "verification_mode": "backend_postcondition",
                                    "visual_audit_pending": True,
                                },
                            )
                        )
        if next_candidate is not None:
            self._publish_interaction_command(next_candidate)
        else:
            self._dispatch(commands)
        if audit_task is not None:
            threading.Thread(
                target=self._run_visual_verification_audit,
                args=audit_task,
                daemon=True,
            ).start()

    def _retry_interaction_approach_after_pose_failure_locked(
        self, payload: dict
    ) -> list[dict]:
        """Retry a bridge-rejected pose through the next preserved approach.

        This method runs while ``self.lock`` is held and intentionally has no
        wall-clock deadline.  The bounded retry count is expressed in actual
        approach options/pose polls so slow rendering or ROS scheduling cannot
        turn a valid interaction into an object-level failure.
        """

        candidate = dict(self.machine.candidate or self.selection or {})
        metadata = candidate.get("metadata") or {}
        two_stage_inner = is_container_two_stage_physical_action(candidate)
        failure_reason = str(
            payload.get("failure_reason") or payload.get("reason") or ""
        ).strip().casefold()
        if (
            failure_reason == "unsafe_open_sweep"
            and str(metadata.get("node_type") or "").casefold() == "container"
            and not two_stage_inner
        ):
            # The bridge has not touched the joint.  Its public result says the
            # current refrigerator stance is unsafe, so the next ring option
            # must earn fresh M1 evidence rather than reusing a face observed
            # at the rejected pose.
            metadata = dict(metadata)
            metadata["observation_required"] = True
            metadata["reobserve"] = True
            metadata["interaction_observation_resolved"] = False
            metadata["interaction_observation_attempts"] = 0
            metadata["unsafe_open_sweep_recommended_retreat_m"] = float(
                payload.get("recommended_retreat_m", 0.0) or 0.0
            )
            candidate["metadata"] = metadata
            self.machine.candidate = candidate
            self.selection = dict(candidate)
        attempts = [
            dict(item)
            for item in metadata.get("interaction_approach_attempts") or []
            if isinstance(item, dict)
        ]
        selected_option_index = max(
            0, int(metadata.get("interaction_approach_goal_option_index", 0) or 0)
        )
        if attempts:
            attempts[-1]["outcome"] = (
                "unsafe_open_sweep"
                if failure_reason == "unsafe_open_sweep"
                else "interaction_pose_invalid"
            )
            attempts[-1]["bridge_pose_validation"] = dict(
                payload.get("interaction_pose_validation") or {}
            )
        else:
            attempts.append(
                {
                    "index": selected_option_index,
                    "goal_xyyaw": list(
                        (metadata.get("effective_interaction_approach_pose_xyyaw")
                        or (candidate.get("interaction_command") or {}).get(
                            "interaction_approach_pose_xyyaw"
                        )
                        or [])
                    ),
                    "outcome": (
                        "unsafe_open_sweep"
                        if failure_reason == "unsafe_open_sweep"
                        else "interaction_pose_invalid"
                    ),
                    "bridge_pose_validation": dict(
                        payload.get("interaction_pose_validation") or {}
                    ),
                }
            )
        goal_option_count = len(navigation_goal_options(candidate))
        approach_attempt_limit = self._interaction_approach_attempt_limit(candidate)
        failure_detail = {
            "reason": (
                "unsafe_open_sweep"
                if failure_reason == "unsafe_open_sweep"
                else "interaction_pose_invalid"
            ),
            "interaction_pose_validation": dict(
                payload.get("interaction_pose_validation") or {}
            ),
        }
        if two_stage_inner:
            # The accepted outer M1 image authorizes all bounded physical
            # points for this one face.  Do not turn a bridge rejection at the
            # primary point into a fresh M1 request until the tangent options
            # have also failed.
            inner_option_count = self._container_two_stage_inner_option_count(candidate)
            next_inner_index = selected_option_index + 1
            if next_inner_index < inner_option_count:
                commands = self.machine.retry_interaction_approach(
                    start_goal_option_index=next_inner_index,
                    interaction_approach_attempts=attempts,
                    detail=failure_detail,
                )
                if commands:
                    self.selection = dict(self.machine.candidate or candidate)
                    return commands
            staging_goals = list(
                metadata.get("container_staging_goal_xyyaw_candidates") or []
            )
            try:
                completed_staging_index = max(
                    0,
                    int(
                        metadata.get(
                            "container_two_stage_staging_goal_option_index", 0
                        )
                    ),
                )
            except (TypeError, ValueError):
                completed_staging_index = len(staging_goals)
            next_staging_index = completed_staging_index + 1
            outer_retry_limit = self._container_two_stage_outer_retry_limit(
                candidate, approach_attempt_limit
            )
            if (
                next_staging_index < outer_retry_limit
            ):
                rospy.logwarn(
                    "[semantic_behavior_executor] bridge rejected inner container "
                    "pose; retrying outer M1 staging option %d/%d",
                    next_staging_index + 1,
                    len(staging_goals),
                )
                commands = self.machine.retry_container_two_stage_staging(
                    next_staging_goal_option_index=next_staging_index,
                    interaction_approach_attempts=attempts,
                    detail=failure_detail,
                )
                if commands:
                    self.selection = dict(self.machine.candidate or candidate)
                    accepted_by_decision = getattr(
                        self, "_container_m1_last_accepted_evidence", None
                    )
                    if isinstance(accepted_by_decision, dict):
                        accepted_by_decision.pop(
                            str(candidate.get("decision_id") or ""), None
                        )
                    return commands
        next_option_index = next_interaction_approach_option_index(
            behavior_type=str(candidate.get("behavior_type") or ""),
            failure_detail=failure_detail,
            selected_option_index=selected_option_index,
            attempted_navigation_count=len(attempts),
            max_navigation_attempts=approach_attempt_limit,
            goal_option_count=goal_option_count,
        )
        if next_option_index is not None:
            rospy.logwarn(
                "[semantic_behavior_executor] bridge pose precondition rejected "
                "INTERACT; retrying approach option %d/%d (attempt %d/%d)",
                next_option_index + 1,
                goal_option_count,
                len(attempts) + 1,
                approach_attempt_limit,
            )
            return self.machine.retry_interaction_approach(
                start_goal_option_index=next_option_index,
                interaction_approach_attempts=attempts,
                detail=failure_detail,
            )
        exhausted_detail = {
            **payload,
            "reason": "interaction_approach_options_exhausted",
            "failure_reason": "interaction_approach_options_exhausted",
            "failure_stage": "interaction_approach_exhausted",
            "interaction_approach_attempts": attempts,
            "interaction_approach_goal_option_count": goal_option_count,
        }
        return self.machine.on_interaction_result(False, detail=exhausted_detail)

    def _retry_interaction_approach_after_visual_gate_locked(
        self, payload: dict
    ) -> list[dict]:
        """Re-observe a container from the next stored approach viewpoint.

        Module 3's side-view rejection happens before the bridge receives an
        action.  It is therefore an approach/viewpoint failure, not evidence
        that the container itself is non-interactable.
        """

        candidate = dict(self.machine.candidate or self.selection or {})
        metadata = candidate.get("metadata") or {}
        attempts = [
            dict(item)
            for item in metadata.get("interaction_approach_attempts") or []
            if isinstance(item, dict)
        ]
        selected_option_index = max(
            0, int(metadata.get("interaction_approach_goal_option_index", 0) or 0)
        )
        visual_plan = dict(payload.get("visual_plan") or {})
        if attempts:
            attempts[-1]["outcome"] = "visual_reposition_required"
            attempts[-1]["visual_gate"] = visual_plan
        else:
            attempts.append(
                {
                    "index": selected_option_index,
                    "goal_xyyaw": list(
                        metadata.get("effective_interaction_approach_pose_xyyaw")
                        or (candidate.get("interaction_command") or {}).get(
                            "interaction_approach_pose_xyyaw"
                        )
                        or []
                    ),
                    "outcome": "visual_reposition_required",
                    "visual_gate": visual_plan,
                }
            )
        goal_option_count = len(navigation_goal_options(candidate))
        approach_attempt_limit = self._interaction_approach_attempt_limit(candidate)
        failure_detail = {
            "reason": "visual_reposition_required",
            "failure_reason": "visual_reposition_required",
            "visual_plan": visual_plan,
            "target_bbox_available": bool(payload.get("target_bbox_available")),
        }
        next_option_index = next_interaction_approach_option_index(
            behavior_type=str(candidate.get("behavior_type") or ""),
            failure_detail=failure_detail,
            selected_option_index=selected_option_index,
            attempted_navigation_count=len(attempts),
            max_navigation_attempts=approach_attempt_limit,
            goal_option_count=goal_option_count,
        )
        self.active_skill_plan = {}
        self.pending_skill_actions = []
        if next_option_index is not None:
            rospy.loginfo(
                "[semantic_behavior_executor] M3 requested container re-observation; "
                "retrying approach option %d/%d (attempt %d/%d)",
                next_option_index + 1,
                goal_option_count,
                len(attempts) + 1,
                approach_attempt_limit,
            )
            return self.machine.retry_interaction_approach(
                start_goal_option_index=next_option_index,
                interaction_approach_attempts=attempts,
                detail=failure_detail,
            )
        exhausted_detail = {
            **payload,
            "reason": "interaction_approach_options_exhausted",
            "failure_reason": "interaction_approach_options_exhausted",
            "failure_stage": "interaction_approach_exhausted",
            "interaction_approach_attempts": attempts,
            "interaction_approach_goal_option_count": goal_option_count,
        }
        return self.machine.on_interaction_result(False, detail=exhausted_detail)

    def _step_sync_callback(self, message: String) -> None:
        """Record a bridge action acknowledgement keyed by evaluator step."""

        try:
            payload = json.loads(message.data)
            step_index = int(payload["step_index"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        with self.lock:
            self._latest_step_sync_index = step_index
            received_at = time.monotonic()
            self._latest_step_sync_received_at = received_at
            action_source = str(payload.get("action_source") or "")
            if self.startup_scan_enabled:
                self._startup_scan_gate.record_step_sync(
                    step_index,
                    action_source=action_source,
                )
            if self.rear_goal_prerotate_step_sync_enabled:
                self._rear_goal_prerotate_gate.record_step_sync(
                    step_index,
                    action_source=action_source,
                )
            if (
                self.interaction_final_align_enabled
                and self.interaction_final_align_step_sync_enabled
            ):
                self._interaction_final_align_gate.record_step_sync(
                    step_index,
                    action_source=action_source,
                )
            monitor = getattr(self, "_rear_dwa_monitor", None)
            latest_cmd = getattr(self, "_latest_cmd_vel_stamped", None)
            if (
                monitor is not None
                and action_source == "cmd_vel"
                and latest_cmd is not None
                and float(latest_cmd.get("received_at", 0.0))
                >= float(monitor.get("started_at", 0.0))
                and received_at - float(latest_cmd.get("received_at", 0.0))
                <= float(getattr(self, "rear_goal_cmd_vel_max_age_s", 0.50))
            ):
                history = monitor.get("samples")
                if isinstance(history, deque) and (
                    not history or int(history[-1].get("step_index", -1)) != step_index
                ):
                    history.append(
                        {
                            "step_index": step_index,
                            "linear_x": float(latest_cmd.get("linear_x", 0.0)),
                            "angular_z": float(latest_cmd.get("angular_z", 0.0)),
                        }
                    )

    def _arm_drawer_scan_execution_wait_locked(self, payload: dict) -> None:
        """Bind a visual-grounded drawer macro to finite simulator progress."""

        if not self.drawer_scan_execution_step_budget_authoritative:
            self._drawer_scan_execution_wait = {}
            return
        self._drawer_scan_execution_wait = {
            "command_id": str(payload.get("command_id") or ""),
            "decision_id": str(payload.get("decision_id") or ""),
            "candidate_id": str(payload.get("candidate_id") or ""),
            "sequence_type": "drawer_scan",
            "started_step_index": self._latest_step_sync_index,
            "started_at_monotonic_s": time.monotonic(),
            "max_task_steps": int(self.drawer_scan_execution_max_task_steps),
            "step_budget_margin": int(self.drawer_scan_execution_step_budget_margin),
        }

    def _clear_drawer_scan_execution_wait_locked(self) -> None:
        self._drawer_scan_execution_wait = {}

    def _drawer_scan_execution_wait_summary_locked(self) -> dict:
        context = dict(getattr(self, "_drawer_scan_execution_wait", {}) or {})
        if not context:
            return {}
        started_step = self._public_step_or_none(context.get("started_step_index"))
        latest_step = self._public_step_or_none(self._latest_step_sync_index)
        if started_step is not None and latest_step is not None:
            context["elapsed_task_steps"] = max(0, int(latest_step) - int(started_step))
        context["latest_step_index"] = latest_step
        return context

    def _drawer_scan_execution_timeout_reason_locked(
        self, now: float | None = None
    ) -> str:
        """Return a bounded drawer-specific timeout reason, or ``""`` to wait.

        The bridge macro advances synchronously with evaluator steps.  As long
        as that finite step stream progresses below its cap, a slow host must
        not invalidate an action that the simulator is still executing.
        """

        context = dict(getattr(self, "_drawer_scan_execution_wait", {}) or {})
        if not context:
            return "interaction_timeout"
        candidate = self.machine.candidate or self.selection or {}
        interaction = dict(candidate.get("interaction_command") or {})
        if (
            self.machine.state != STATE_INTERACTING
            or str(interaction.get("sequence_type") or "").casefold()
            != "drawer_scan"
            or str(context.get("decision_id") or "")
            != str(candidate.get("decision_id") or "")
        ):
            self._clear_drawer_scan_execution_wait_locked()
            return "interaction_timeout"
        now = time.monotonic() if now is None else float(now)
        started_at = float(context.get("started_at_monotonic_s", now) or now)
        if now - started_at > self.drawer_scan_execution_wall_cap_s:
            return "drawer_scan_execution_wall_cap"
        started_step = self._public_step_or_none(context.get("started_step_index"))
        latest_step = self._public_step_or_none(self._latest_step_sync_index)
        if started_step is None or latest_step is None:
            # Do not extend the generic timeout without a public evaluator-step
            # clock.  This preserves a bounded failure for a missing bridge.
            return "interaction_timeout"
        if latest_step <= started_step:
            if now - started_at >= self.drawer_scan_execution_step_sync_stall_timeout_s:
                return "drawer_scan_execution_step_sync_stall"
            return ""
        if (
            now - float(self._latest_step_sync_received_at or started_at)
            >= self.drawer_scan_execution_step_sync_stall_timeout_s
        ):
            return "drawer_scan_execution_step_sync_stall"
        elapsed_steps = int(latest_step) - int(started_step)
        allowed_steps = int(context.get("max_task_steps", 0) or 0) + int(
            context.get("step_budget_margin", 0) or 0
        )
        if elapsed_steps > max(1, allowed_steps):
            return "drawer_scan_execution_step_budget_exhausted"
        return ""

    def _effective_timeout_reason_locked(self, now: float | None = None) -> str:
        reason = self.machine.timeout_reason(now=now)
        if reason != "interaction_timeout":
            return reason
        # A finite, advancing drawer macro owns its timeout in simulator steps.
        # Every other interaction retains the existing wall-clock behavior.
        return self._drawer_scan_execution_timeout_reason_locked(now=now)

    def _fresh_command_gate_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            step_index = int(payload["step_index"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        with self.lock:
            if self.startup_scan_enabled:
                self._startup_scan_gate.record_fresh_gate(step_index)
            if self.rear_goal_prerotate_step_sync_enabled:
                self._rear_goal_prerotate_gate.record_fresh_gate(step_index)
            if (
                self.interaction_final_align_enabled
                and self.interaction_final_align_step_sync_enabled
            ):
                self._interaction_final_align_gate.record_fresh_gate(step_index)

    def _image_callback(self, message: Image) -> None:
        try:
            image = self._decode_ros_image(message)
        except Exception:
            return
        try:
            rgb_step_seq = int(message.header.seq)
        except (AttributeError, TypeError, ValueError):
            rgb_step_seq = None
        received_at = time.monotonic()
        with self.lock:
            self.latest_image = image.copy()
            self.latest_image_sequence += 1
            if rgb_step_seq is not None:
                self._latest_rgb_step_seq = rgb_step_seq
                self._latest_rgb_step_received_at = received_at
                if self.startup_scan_enabled:
                    self._startup_scan_gate.record_rgb(rgb_step_seq, now=received_at)
                if self.rear_goal_prerotate_step_sync_enabled:
                    self._rear_goal_prerotate_gate.record_rgb(
                        rgb_step_seq, now=received_at
                    )
                if (
                    self.interaction_final_align_enabled
                    and self.interaction_final_align_step_sync_enabled
                ):
                    self._interaction_final_align_gate.record_rgb(
                        rgb_step_seq, now=received_at
                    )

    @staticmethod
    def _decode_ros_image(message: Image):
        channels_by_encoding = {
            "bgr8": 3,
            "rgb8": 3,
            "bgra8": 4,
            "rgba8": 4,
        }
        encoding = str(message.encoding or "").casefold()
        channels = channels_by_encoding.get(encoding)
        if channels is None or message.height <= 0 or message.width <= 0:
            raise ValueError(f"unsupported image encoding: {message.encoding}")
        row_width = int(message.step or message.width * channels)
        raw = np.frombuffer(message.data, dtype=np.uint8).reshape(
            int(message.height), row_width
        )
        image = raw[:, : int(message.width) * channels].reshape(
            int(message.height), int(message.width), channels
        )
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def _graph_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            self.latest_graph = payload
            commands = (
                self._verify_graph_locked()
                if requires_graph_verification(
                    self.ablation.module3, self.selection
                )
                else []
            )
        self._dispatch(commands)

    def _occupancy_callback(self, message: OccupancyGrid) -> None:
        with self.lock:
            self._latest_occupancy = message
            self._latest_occupancy_received_at = time.monotonic()

    def _cmd_vel_stamped_callback(self, message: TwistStamped) -> None:
        """Keep the latest relay command for step-indexed DWA diagnostics."""

        try:
            twist = message.twist
            linear_x = float(twist.linear.x)
            angular_z = float(twist.angular.z)
        except (AttributeError, TypeError, ValueError):
            return
        with self.lock:
            self._latest_cmd_vel_stamped = {
                "linear_x": linear_x,
                "angular_z": angular_z,
                "received_at": time.monotonic(),
            }

    @staticmethod
    def _map_header_fields(
        message: OccupancyGrid | OccupancyGridUpdate,
    ) -> tuple[int | None, float | None]:
        """Extract diagnostic map-header values without trusting either one.

        Local receipt counters carry the synchronization guarantee.  Header
        stamps additionally prove that a raw OCC was generated after the
        evaluator's action result when both publisher clocks are available.
        """

        header = getattr(message, "header", None)
        try:
            header_seq = int(getattr(header, "seq", None))
        except (TypeError, ValueError):
            header_seq = None
        stamp = getattr(header, "stamp", None)
        try:
            stamp_sec = float(stamp.to_sec())
        except (AttributeError, TypeError, ValueError):
            try:
                stamp_sec = float(stamp.secs) + float(stamp.nsecs) * 1e-9
            except (AttributeError, TypeError, ValueError):
                stamp_sec = None
        if stamp_sec is not None and not math.isfinite(stamp_sec):
            stamp_sec = None
        return header_seq, stamp_sec

    def _raw_occupancy_callback(self, message: OccupancyGrid) -> None:
        """Record raw SLAM OCC and the planning counter visible at receipt."""

        header_seq, header_stamp_sec = self._map_header_fields(message)
        with self._global_costmap_condition:
            self._raw_occupancy_received_count += 1
            self._latest_raw_occupancy_header_seq = header_seq
            self._latest_raw_occupancy_header_stamp_sec = header_stamp_sec
            self._latest_raw_occupancy_received_at = time.monotonic()
            self._raw_occupancy_events.append(
                PostInteractionRawMapBarrier(
                    receipt_count=self._raw_occupancy_received_count,
                    header_seq=header_seq,
                    header_stamp_sec=header_stamp_sec,
                    planning_occupancy_receipt_count=(
                        self._planning_occupancy_received_count
                    ),
                    planning_occupancy_header_seq=(
                        self._latest_planning_occupancy_header_seq
                    ),
                    planning_occupancy_header_stamp_sec=(
                        self._latest_planning_occupancy_header_stamp_sec
                    ),
                )
            )
            self._global_costmap_condition.notify_all()

    def _planning_occupancy_callback(self, message: OccupancyGrid) -> None:
        """Record the semantic OCC actually consumed by StaticLayer."""

        header_seq, header_stamp_sec = self._map_header_fields(message)
        with self._global_costmap_condition:
            self._planning_occupancy_received_count += 1
            self._latest_planning_occupancy_header_seq = header_seq
            self._latest_planning_occupancy_header_stamp_sec = header_stamp_sec
            self._latest_planning_occupancy_received_at = time.monotonic()
            self._global_costmap_condition.notify_all()

    def _global_costmap_callback(self, message: OccupancyGrid) -> None:
        """Record a full global-costmap fallback publication."""

        header_seq, _ = self._map_header_fields(message)
        with self._global_costmap_condition:
            self._global_costmap_received_count += 1
            self._latest_global_costmap_header_seq = header_seq
            self._latest_global_costmap_received_at = time.monotonic()
            self._global_costmap_condition.notify_all()

    def _global_costmap_update_callback(self, message: OccupancyGridUpdate) -> None:
        """Record the primary incremental global-costmap planner update."""

        header_seq, _ = self._map_header_fields(message)
        with self._global_costmap_condition:
            self._global_costmap_update_received_count += 1
            self._latest_global_costmap_update_header_seq = header_seq
            self._latest_global_costmap_update_received_at = time.monotonic()
            self._global_costmap_condition.notify_all()

    def _record_post_interaction_costmap_baseline_locked(
        self, payload: dict
    ) -> None:
        """Bind a successful portal-open result to every map-stage baseline.

        A semantic graph revision and a global-costmap delta can both precede
        the physical map change.  The matching traversal must observe the
        causal chain raw OCC -> planning OCC -> global costmap after this
        exact result before it asks ``make_plan``.
        """

        # A static portal reports ``static_open`` synchronously but does not
        # move geometry.  Only an articulated open can establish the causal
        # raw OCC -> planning OCC -> costmap barrier for a forced traversal.
        if is_static_portal_interaction_feedback(self.selection, payload):
            return
        if not (
            payload.get("success") is True
            or str(payload.get("status") or "").upper() == "SUCCEEDED"
        ):
            return
        selection = self.selection or {}
        interaction = selection.get("interaction_command") or {}
        metadata = selection.get("metadata") or {}
        detail = payload.get("detail") or {}
        node_type = str(
            payload.get("node_type")
            or detail.get("node_type")
            or interaction.get("node_type")
            or metadata.get("node_type")
            or ""
        ).casefold()
        action = str(
            payload.get("action")
            or detail.get("action")
            or interaction.get("action")
            or ""
        ).casefold()
        if node_type != "portal" or action != "open":
            return
        portal_id = str(
            payload.get("node_id")
            or detail.get("node_id")
            or interaction.get("node_id")
            or selection.get("target_id")
            or ""
        )
        source_event_id = str(
            payload.get("event_id")
            or detail.get("event_id")
            or selection.get("decision_id")
            or ""
        )
        keys = post_interaction_costmap_baseline_keys(source_event_id, portal_id)
        if not keys:
            return
        try:
            result_stamp_sec = float(payload.get("stamp_sec"))
        except (TypeError, ValueError):
            result_stamp_sec = None
        if result_stamp_sec is not None and not math.isfinite(result_stamp_sec):
            result_stamp_sec = None
        baseline = PostInteractionCostmapBaseline(
            portal_id=portal_id,
            source_event_id=source_event_id,
            receipt_count=self._global_costmap_received_count,
            header_seq=self._latest_global_costmap_header_seq,
            update_receipt_count=self._global_costmap_update_received_count,
            update_header_seq=self._latest_global_costmap_update_header_seq,
            raw_occupancy_receipt_count=self._raw_occupancy_received_count,
            raw_occupancy_header_seq=self._latest_raw_occupancy_header_seq,
            raw_occupancy_header_stamp_sec=(
                self._latest_raw_occupancy_header_stamp_sec
            ),
            planning_occupancy_receipt_count=(
                self._planning_occupancy_received_count
            ),
            planning_occupancy_header_seq=(
                self._latest_planning_occupancy_header_seq
            ),
            planning_occupancy_header_stamp_sec=(
                self._latest_planning_occupancy_header_stamp_sec
            ),
            interaction_result_stamp_sec=result_stamp_sec,
        )
        for key in keys:
            self._post_interaction_costmap_baselines[key] = baseline
            self._post_interaction_raw_map_barriers.pop(key, None)
            self._post_interaction_planning_map_barriers.pop(key, None)
        rospy.loginfo(
            "[semantic_behavior_executor] recorded post-open map baseline: "
            "portal=%s event=%s result_stamp=%s raw=%d planning=%d "
            "update=%d full=%d",
            portal_id,
            source_event_id,
            str(baseline.interaction_result_stamp_sec),
            baseline.raw_occupancy_receipt_count,
            baseline.planning_occupancy_receipt_count,
            baseline.update_receipt_count,
            baseline.receipt_count,
        )

    def _post_interaction_costmap_baseline_locked(
        self, candidate: dict
    ) -> tuple[PostInteractionCostmapBaseline | None, str]:
        metadata = candidate.get("metadata") or {}
        portal_id = str(
            metadata.get("opened_portal_id") or candidate.get("target_id") or ""
        )
        source_event_id = str(metadata.get("source_interaction_event_id") or "")
        for key in post_interaction_costmap_baseline_keys(
            source_event_id, portal_id
        ):
            baseline = self._post_interaction_costmap_baselines.get(key)
            if baseline is not None:
                return baseline, key
        return None, ""

    def _post_interaction_raw_map_barrier_locked(
        self,
        baseline: PostInteractionCostmapBaseline,
        baseline_key: str,
    ) -> tuple[PostInteractionRawMapBarrier | None, str]:
        """Choose the first raw OCC receipt proven newer than the open."""

        recorded = self._post_interaction_raw_map_barriers.get(baseline_key)
        if recorded is not None:
            return recorded
        for raw_event in self._raw_occupancy_events:
            raw_fresh_source = post_interaction_raw_occupancy_fresh_source(
                baseline,
                raw_event.receipt_count,
                raw_event.header_stamp_sec,
            )
            if not raw_fresh_source:
                continue
            recorded = raw_event, raw_fresh_source
            self._post_interaction_raw_map_barriers[baseline_key] = recorded
            rospy.loginfo(
                "[semantic_behavior_executor] admitted post-open raw OCC: "
                "portal=%s source=%s result_stamp=%s raw_receipt=%d "
                "raw_stamp=%s planning_baseline=%d",
                baseline.portal_id,
                raw_fresh_source,
                str(baseline.interaction_result_stamp_sec),
                raw_event.receipt_count,
                str(raw_event.header_stamp_sec),
                raw_event.planning_occupancy_receipt_count,
            )
            return recorded
        return None, ""

    def _post_interaction_planning_map_barrier_locked(
        self,
        baseline_key: str,
        raw_barrier: PostInteractionRawMapBarrier,
        raw_fresh_source: str,
    ) -> PostInteractionPlanningMapBarrier | None:
        """Record planning OCC, then snapshot the costmap counters after it."""

        existing = self._post_interaction_planning_map_barriers.get(baseline_key)
        if existing is not None:
            return existing
        planning_fresh_source = post_interaction_planning_occupancy_fresh_source(
            raw_barrier,
            self._planning_occupancy_received_count,
            self._latest_planning_occupancy_header_stamp_sec,
        )
        if not planning_fresh_source:
            return None
        barrier = PostInteractionPlanningMapBarrier(
            raw_map=raw_barrier,
            raw_fresh_source=raw_fresh_source,
            receipt_count=self._planning_occupancy_received_count,
            header_seq=self._latest_planning_occupancy_header_seq,
            header_stamp_sec=self._latest_planning_occupancy_header_stamp_sec,
            planning_fresh_source=planning_fresh_source,
            costmap_receipt_count=self._global_costmap_received_count,
            costmap_header_seq=self._latest_global_costmap_header_seq,
            costmap_update_receipt_count=(
                self._global_costmap_update_received_count
            ),
            costmap_update_header_seq=(
                self._latest_global_costmap_update_header_seq
            ),
        )
        self._post_interaction_planning_map_barriers[baseline_key] = barrier
        rospy.loginfo(
            "[semantic_behavior_executor] admitted post-open planning OCC: "
            "source=%s receipt=%d stamp=%s; awaiting later global costmap "
            "update=%d full=%d",
            planning_fresh_source,
            barrier.receipt_count,
            str(barrier.header_stamp_sec),
            barrier.costmap_update_receipt_count,
            barrier.costmap_receipt_count,
        )
        return barrier

    def _wait_for_post_interaction_costmap_freshness(
        self, decision_id: str, candidate: dict
    ) -> tuple[bool, dict]:
        """Hold traversal through raw OCC -> planning OCC -> costmap.

        A costmap delta immediately after an interaction can describe a
        pre-open planning map.  Do not call ``make_plan`` until a raw SLAM map
        newer than the action result has reached the semantic planner and a
        later global-costmap delta (or full map) has followed it.  The entire
        causal chain shares one bounded wait budget.
        """

        started_at = time.monotonic()
        deadline = started_at + self.post_interaction_costmap_fresh_timeout_s
        with self._global_costmap_condition:
            baseline, baseline_key = self._post_interaction_costmap_baseline_locked(
                candidate
            )
            while True:
                current_full_count = self._global_costmap_received_count
                current_full_header_seq = self._latest_global_costmap_header_seq
                current_update_count = self._global_costmap_update_received_count
                current_update_header_seq = (
                    self._latest_global_costmap_update_header_seq
                )
                current_raw_count = self._raw_occupancy_received_count
                current_raw_header_seq = self._latest_raw_occupancy_header_seq
                current_raw_header_stamp_sec = (
                    self._latest_raw_occupancy_header_stamp_sec
                )
                current_planning_count = self._planning_occupancy_received_count
                current_planning_header_seq = (
                    self._latest_planning_occupancy_header_seq
                )
                current_planning_header_stamp_sec = (
                    self._latest_planning_occupancy_header_stamp_sec
                )
                raw_barrier: PostInteractionRawMapBarrier | None = None
                raw_fresh_source = ""
                planning_barrier: PostInteractionPlanningMapBarrier | None = None
                if baseline is not None:
                    raw_barrier, raw_fresh_source = (
                        self._post_interaction_raw_map_barrier_locked(
                            baseline, baseline_key
                        )
                    )
                    if raw_barrier is not None:
                        planning_barrier = (
                            self._post_interaction_planning_map_barrier_locked(
                                baseline_key,
                                raw_barrier,
                                raw_fresh_source,
                            )
                        )
                if raw_barrier is None:
                    causal_stage = "waiting_raw_occupancy"
                elif planning_barrier is None:
                    causal_stage = "waiting_planning_occupancy"
                else:
                    causal_stage = "waiting_global_costmap"
                costmap_baseline_full_count = (
                    None
                    if baseline is None
                    else (
                        planning_barrier.costmap_receipt_count
                        if planning_barrier is not None
                        else baseline.receipt_count
                    )
                )
                costmap_baseline_full_header_seq = (
                    None
                    if baseline is None
                    else (
                        planning_barrier.costmap_header_seq
                        if planning_barrier is not None
                        else baseline.header_seq
                    )
                )
                costmap_baseline_update_count = (
                    None
                    if baseline is None
                    else (
                        planning_barrier.costmap_update_receipt_count
                        if planning_barrier is not None
                        else baseline.update_receipt_count
                    )
                )
                costmap_baseline_update_header_seq = (
                    None
                    if baseline is None
                    else (
                        planning_barrier.costmap_update_header_seq
                        if planning_barrier is not None
                        else baseline.update_header_seq
                    )
                )
                elapsed_s = max(0.0, time.monotonic() - started_at)
                detail = {
                    "opened_portal_id": str(
                        (candidate.get("metadata") or {}).get("opened_portal_id")
                        or candidate.get("target_id")
                        or ""
                    ),
                    "post_open_costmap_baseline_key": baseline_key,
                    "post_open_costmap_baseline_receipt_count": (
                        costmap_baseline_full_count
                    ),
                    "post_open_costmap_baseline_header_seq": (
                        costmap_baseline_full_header_seq
                    ),
                    "post_open_costmap_latest_receipt_count": current_full_count,
                    "post_open_costmap_latest_header_seq": current_full_header_seq,
                    "post_open_costmap_baseline_update_receipt_count": (
                        costmap_baseline_update_count
                    ),
                    "post_open_costmap_baseline_update_header_seq": (
                        costmap_baseline_update_header_seq
                    ),
                    "post_open_costmap_latest_update_receipt_count": (
                        current_update_count
                    ),
                    "post_open_costmap_latest_update_header_seq": (
                        current_update_header_seq
                    ),
                    "post_open_costmap_wait_elapsed_s": elapsed_s,
                    "post_open_costmap_wait_timeout_s": (
                        self.post_interaction_costmap_fresh_timeout_s
                    ),
                    "post_open_causal_map_stage": causal_stage,
                    "post_open_result_stamp_sec": (
                        None
                        if baseline is None
                        else baseline.interaction_result_stamp_sec
                    ),
                    "post_open_raw_occ_baseline_receipt_count": (
                        None
                        if baseline is None
                        else baseline.raw_occupancy_receipt_count
                    ),
                    "post_open_raw_occ_baseline_header_seq": (
                        None
                        if baseline is None
                        else baseline.raw_occupancy_header_seq
                    ),
                    "post_open_raw_occ_baseline_header_stamp_sec": (
                        None
                        if baseline is None
                        else baseline.raw_occupancy_header_stamp_sec
                    ),
                    "post_open_raw_occ_latest_receipt_count": current_raw_count,
                    "post_open_raw_occ_latest_header_seq": current_raw_header_seq,
                    "post_open_raw_occ_latest_header_stamp_sec": (
                        current_raw_header_stamp_sec
                    ),
                    "post_open_raw_occ_admitted_receipt_count": (
                        None if raw_barrier is None else raw_barrier.receipt_count
                    ),
                    "post_open_raw_occ_admitted_header_seq": (
                        None if raw_barrier is None else raw_barrier.header_seq
                    ),
                    "post_open_raw_occ_admitted_header_stamp_sec": (
                        None
                        if raw_barrier is None
                        else raw_barrier.header_stamp_sec
                    ),
                    "post_open_raw_occ_fresh_source": raw_fresh_source,
                    "post_open_planning_occ_baseline_receipt_count": (
                        None
                        if baseline is None
                        else baseline.planning_occupancy_receipt_count
                    ),
                    "post_open_planning_occ_baseline_header_seq": (
                        None
                        if baseline is None
                        else baseline.planning_occupancy_header_seq
                    ),
                    "post_open_planning_occ_baseline_header_stamp_sec": (
                        None
                        if baseline is None
                        else baseline.planning_occupancy_header_stamp_sec
                    ),
                    "post_open_planning_occ_latest_receipt_count": (
                        current_planning_count
                    ),
                    "post_open_planning_occ_latest_header_seq": (
                        current_planning_header_seq
                    ),
                    "post_open_planning_occ_latest_header_stamp_sec": (
                        current_planning_header_stamp_sec
                    ),
                    "post_open_planning_occ_admitted_receipt_count": (
                        None
                        if planning_barrier is None
                        else planning_barrier.receipt_count
                    ),
                    "post_open_planning_occ_admitted_header_seq": (
                        None
                        if planning_barrier is None
                        else planning_barrier.header_seq
                    ),
                    "post_open_planning_occ_admitted_header_stamp_sec": (
                        None
                        if planning_barrier is None
                        else planning_barrier.header_stamp_sec
                    ),
                    "post_open_planning_occ_fresh_source": (
                        ""
                        if planning_barrier is None
                        else planning_barrier.planning_fresh_source
                    ),
                    "post_open_causal_costmap_baseline_receipt_count": (
                        costmap_baseline_full_count
                    ),
                    "post_open_causal_costmap_baseline_update_receipt_count": (
                        costmap_baseline_update_count
                    ),
                    # Stable primary trace keys are local incremental-update
                    # receipt sequences; after planning OCC they are sampled
                    # at that causal barrier instead of at action result.
                    "baseline_global_costmap_seq": (
                        costmap_baseline_update_count
                    ),
                    "observed_global_costmap_seq": current_update_count,
                    "baseline_global_costmap_header_seq": (
                        costmap_baseline_update_header_seq
                    ),
                    "observed_global_costmap_header_seq": current_update_header_seq,
                    "baseline_global_costmap_full_seq": (
                        costmap_baseline_full_count
                    ),
                    "observed_global_costmap_full_seq": current_full_count,
                    "baseline_global_costmap_full_header_seq": (
                        costmap_baseline_full_header_seq
                    ),
                    "observed_global_costmap_full_header_seq": (
                        current_full_header_seq
                    ),
                    "costmap_wait_elapsed": elapsed_s,
                    "costmap_fresh": False,
                    "fresh_source": "",
                }
                if baseline is None:
                    detail["reason"] = "post_open_costmap_baseline_missing"
                    rospy.logwarn(
                        "[semantic_behavior_executor] post-open traversal has "
                        "no global-costmap baseline: portal=%s",
                        detail["opened_portal_id"],
                    )
                    return False, detail
                fresh_source = ""
                if planning_barrier is not None:
                    fresh_source = post_interaction_costmap_receipts_fresh_source(
                        planning_barrier.costmap_receipt_count,
                        planning_barrier.costmap_update_receipt_count,
                        current_full_count,
                        current_update_count,
                    )
                if fresh_source:
                    detail["post_open_costmap_fresh"] = True
                    detail["costmap_fresh"] = True
                    detail["fresh_source"] = fresh_source
                    detail["post_open_causal_map_stage"] = "ready"
                    rospy.loginfo(
                        "[semantic_behavior_executor] causal post-open map "
                        "chain admits preflight: portal=%s source=%s raw=%d "
                        "planning=%d update=%d->%d full=%d->%d wait=%.3fs",
                        baseline.portal_id,
                        fresh_source,
                        raw_barrier.receipt_count if raw_barrier is not None else -1,
                        planning_barrier.receipt_count,
                        planning_barrier.costmap_update_receipt_count,
                        current_update_count,
                        planning_barrier.costmap_receipt_count,
                        current_full_count,
                        elapsed_s,
                    )
                    return True, detail
                if not (
                    self.selection is not None
                    and str(self.selection.get("decision_id") or "") == decision_id
                    and self.machine.state
                    in {STATE_NAVIGATING, STATE_APPROACH_INTERACTION}
                ):
                    detail["reason"] = "post_open_costmap_wait_preempted"
                    return False, detail
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    detail["reason"] = {
                        "waiting_raw_occupancy": "post_open_raw_occ_refresh_timeout",
                        "waiting_planning_occupancy": (
                            "post_open_planning_occ_refresh_timeout"
                        ),
                    }.get(causal_stage, "post_open_costmap_refresh_timeout")
                    detail["post_open_costmap_fresh"] = False
                    rospy.logwarn(
                        "[semantic_behavior_executor] post-open causal-map "
                        "timeout: portal=%s stage=%s raw=%d planning=%d "
                        "update=%s->%d full=%s->%d wait=%.3fs",
                        baseline.portal_id,
                        causal_stage,
                        current_raw_count,
                        current_planning_count,
                        str(costmap_baseline_update_count),
                        current_update_count,
                        str(costmap_baseline_full_count),
                        current_full_count,
                        elapsed_s,
                    )
                    return False, detail
                self._global_costmap_condition.wait(
                    timeout=min(
                        remaining_s,
                        self.post_interaction_costmap_fresh_poll_interval_s,
                    )
                )

    def _local_plan_callback(self, message: Path) -> None:
        # A non-empty plan is only a watchdog hint.  It is additionally scoped
        # to the current navigation send time below, so a retained plan from a
        # prior goal cannot suppress a genuine stall.
        with self.lock:
            self._latest_local_plan_received_at = time.monotonic()
            self._latest_local_plan_pose_count = len(message.poses or [])

    def _verify_graph_locked(self) -> list[dict]:
        if self.machine.state != STATE_VERIFYING or self.selection is None:
            return []
        target_id = str(self.selection.get("target_id") or "")
        target_name = str(self.selection.get("target_name") or "")
        for node in self.latest_graph.get("nodes") or []:
            attributes = node.get("attributes") or {}
            if str(node.get("id") or "") != target_id and str(
                attributes.get("source_object_name") or node.get("name") or ""
            ) != target_name:
                continue
            metadata = self.selection.get("metadata") or {}
            if (
                str(self.selection.get("behavior_type") or "") == "NAVIGATE"
                and bool(metadata.get("target_goal"))
            ):
                visible_pixels = int(attributes.get("visible_pixels", 0) or 0)
                min_visible_pixels = int(
                    metadata.get("target_min_visible_pixels", 1) or 1
                )
                target_visible = bool(node.get("is_currently_visible")) and (
                    visible_pixels >= min_visible_pixels
                )
                visible_fraction = float(
                    attributes.get("visible_fraction", 1.0) or 0.0
                )
                consecutive_observations = int(
                    attributes.get("consecutive_observations", 2) or 0
                )
                target_visible = target_visible and (
                    visible_fraction >= float(
                        metadata.get("target_min_visible_fraction", 0.2) or 0.2
                    )
                    and consecutive_observations >= int(
                        metadata.get("target_min_consecutive_observations", 2) or 2
                    )
                )
                return self.machine.on_target_visibility(
                    target_visible,
                    detail={
                        "node_id": node.get("id"),
                        "target_visible": target_visible,
                        "visible_pixels": visible_pixels,
                        "visible_fraction": visible_fraction,
                        "consecutive_observations": consecutive_observations,
                        "min_visible_pixels": min_visible_pixels,
                        "graph_revision": self.latest_graph.get(
                            "graph_revision", 0
                        ),
                    },
                )
            interaction = node.get("interaction") or {}
            return self.machine.on_graph_state(
                str(interaction.get("state") or "unknown"),
                detail={
                    "node_id": node.get("id"),
                    "state": interaction.get("state"),
                    "graph_revision": self.latest_graph.get("graph_revision", 0),
                },
            )
        return []

    def _tick(self, _event) -> None:
        reservation_retry = None
        with self.lock:
            reason = self._effective_timeout_reason_locked()
            if reason in {"navigation_timeout", "interaction_navigation_timeout"}:
                reason = ""
            cancel_navigation = bool(reason) and self.machine.state in {
                STATE_NAVIGATING,
                STATE_APPROACH_INTERACTION,
            }
            if reason == "scan_timeout":
                detail = dict(self._startup_scan_progress)
                detail["reason"] = "scan_timeout"
                commands = self.machine.on_scan_result(False, detail)
            elif reason == "drawer_scan_fresh_frame_timeout":
                decision_id = str((self.selection or {}).get("decision_id") or "")
                detail = self._drawer_scan_wait_detail_locked(
                    decision_id,
                    status="timeout",
                    reason="drawer_scan_fresh_frame_timeout",
                )
                detail["reason"] = "drawer_scan_fresh_frame_timeout"
                if decision_id:
                    self._drawer_scan_wait_records[decision_id] = dict(detail)
                    self._drawer_scan_wait_contexts.pop(decision_id, None)
                commands = self.machine.on_drawer_scan_wait_failed(detail)
            else:
                commands = self.machine.fail_timeout(reason) if reason else []
            if (
                not reason
                and self.machine.state == STATE_PREPARING_EXPLORE
                and self.selection is not None
                and self.explore_reservation_retry_sec > 0.0
            ):
                now = time.monotonic()
                if now - self._last_explore_reservation_publish_at >= self.explore_reservation_retry_sec:
                    reservation_retry = dict(self.selection)
            state_payload = {
                **self.machine.summary(),
                "decision_id": "" if self.selection is None else self.selection.get("decision_id", ""),
                "explore_reservation_publish_count": self._explore_reservation_publish_count,
                "explore_reservation_waiting_for_ack": self.machine.state
                == STATE_PREPARING_EXPLORE,
                "explore_feedback_received_count": self._explore_feedback_received_count,
                "explore_feedback_matched_count": self._explore_feedback_matched_count,
                "explore_feedback_ignored_count": self._explore_feedback_ignored_count,
                "last_explore_feedback": dict(self._last_explore_feedback),
                "external_recovery": {
                    "enabled": self.external_recovery_request_enabled,
                    "pending_count": len(self._external_recovery_requests),
                    "received_count": self._external_recovery_received_count,
                    "accepted_count": self._external_recovery_accepted_count,
                    "rejected_count": self._external_recovery_rejected_count,
                    "consumed_count": self._external_recovery_consumed_count,
                    "last_feedback": dict(self._external_recovery_last_feedback),
                },
                "drawer_scan_wait": dict(
                    self._drawer_scan_wait_contexts.get(
                        str((self.selection or {}).get("decision_id") or ""),
                        {},
                    )
                ),
                "drawer_scan_execution_wait": self._drawer_scan_execution_wait_summary_locked(),
                "startup_scan": dict(self._startup_scan_progress),
                "post_interaction_visual_audit_count": len(
                    getattr(self, "_post_interaction_visual_audits", ())
                ),
                "post_interaction_visual_audit_last": (
                    dict(getattr(self, "_post_interaction_visual_audits", ())[-1])
                    if getattr(self, "_post_interaction_visual_audits", ())
                    else {}
                ),
                "timestamp": time.time(),
            }
        self.state_pub.publish(
            String(data=json.dumps(state_payload, ensure_ascii=False, separators=(",", ":")))
        )
        if cancel_navigation:
            self.move_base.cancel_goal()
        self._dispatch(commands)
        if reservation_retry is not None:
            self._publish_explore_command(reservation_retry, action="reserve_frontier")

    def _dispatch(self, commands: list[dict]) -> None:
        for command in commands:
            kind = command.get("kind")
            if kind == "reserve_frontier":
                self._publish_explore_command(
                    command["candidate"], action="reserve_frontier"
                )
            elif kind == "finalize_frontier":
                self._publish_explore_command(
                    command["candidate"],
                    action="finalize_frontier",
                    success=bool(command.get("success")),
                    detail=command.get("detail") or {},
                )
            elif kind == "scan":
                candidate = dict(command["candidate"])
                decision_id = str(candidate.get("decision_id") or "")
                threading.Thread(
                    target=self._run_startup_scan,
                    args=(decision_id, candidate),
                    daemon=True,
                ).start()
            elif kind == "navigate":
                candidate = dict(command["candidate"])
                decision_id = str(candidate.get("decision_id") or "")
                threading.Thread(
                    target=self._run_navigation,
                    args=(
                        decision_id,
                        candidate,
                        int(command.get("start_goal_option_index", 0) or 0),
                        list(command.get("interaction_approach_attempts") or []),
                    ),
                    daemon=True,
                ).start()
            elif kind == "interact":
                # M1 owns the pre-interaction visual state and approach
                # readiness. M3 is strictly a post-action audit, so it must
                # never delay or reinterpret this command before execution.
                self._publish_interaction_command(command["candidate"])
            elif kind == "request_interaction_observation":
                self._publish_interaction_observation_request(command)
            elif kind == "wait_for_drawer_scan":
                candidate = dict(command["candidate"])
                decision_id = str(candidate.get("decision_id") or "")
                threading.Thread(
                    target=self._wait_for_fresh_drawer_scan,
                    args=(decision_id, candidate),
                    daemon=True,
                ).start()
            elif kind == "publish_drawer_scan":
                self._publish_interaction_command(command["candidate"])
            elif kind == "terminal":
                self._finish_terminal(command)

    def _publish_interaction_observation_request(self, command: dict) -> None:
        """Ask M1 for a causally later portal or drawer visual view."""

        candidate = dict(command.get("candidate") or {})
        decision_id = str(candidate.get("decision_id") or "")
        if not decision_id:
            return
        with self.lock:
            if (
                self.selection is None
                or str(self.selection.get("decision_id") or "") != decision_id
                or self.machine.state != STATE_WAITING_FOR_INTERACTION_OBSERVATION
            ):
                return
            interaction = candidate.get("interaction_command") or {}
            object_id = str(
                command.get("object_id")
                or interaction.get("object_id")
                or candidate.get("target_id")
                or ""
            )
            node_id = str(command.get("node_id") or candidate.get("target_id") or "")
            if not object_id:
                return
            minimum_capture_step = command.get("min_capture_step")
            try:
                minimum_capture_step = int(minimum_capture_step)
            except (TypeError, ValueError):
                minimum_capture_step = self._public_step_or_none(
                    (self.latest_graph or {}).get("capture_step")
                )
            if minimum_capture_step is None:
                minimum_capture_step = max(0, int(self._latest_rgb_step_seq or 0))
            if self.machine.candidate is not None:
                metadata = dict(self.machine.candidate.get("metadata") or {})
                metadata["interaction_observation_min_capture_step"] = int(
                    minimum_capture_step
                )
                self.machine.candidate["metadata"] = metadata
            self.interaction_observation_sequence += 1
            request_id = (
                f"{decision_id}:m1:{self.interaction_observation_sequence:03d}"
            )[:96]
            episode_id = str(
                candidate.get("episode_id")
                or self.selection.get("episode_id")
                or (self.latest_graph or {}).get("episode_id")
                or ""
            )
            request = {
                "object_id": object_id,
                "episode_id": episode_id,
                "minimum_capture_step": int(minimum_capture_step),
                "reason": str(
                    command.get("reason")
                    or "mllm_portal_state_unknown"
                )[:160],
                "request_id": request_id,
            }
            if self._is_container_pre_action_candidate(self.machine.candidate):
                # Public graph/candidate semantics constrain only the M1 class
                # contract for this targeted re-observation.  M1 still decides
                # visual state and frontality from the new RGB image itself.
                request["expected_node_type"] = "container"
            metadata = (
                dict(self.machine.candidate.get("metadata") or {})
                if self.machine.candidate
                else {}
            )
            interaction_pose = list(
                metadata.get("effective_interaction_approach_pose_xyyaw")
                or interaction.get("interaction_approach_pose_xyyaw")
                or (self.machine.candidate or {}).get("goal_xyyaw")
                or []
            )
            self._interaction_observation_requests[decision_id] = {
                **request,
                "node_id": node_id,
                "attempt": int(command.get("attempt", 0) or 0),
                "confirmation_count": max(
                    0, int(command.get("confirmation_count", 0) or 0)
                ),
                "drawer_pre_action": self._is_drawer_pre_action_candidate(
                    self.machine.candidate
                ),
                "container_pre_action": self._is_container_pre_action_candidate(
                    self.machine.candidate
                ),
                "observation_pose_xyyaw": interaction_pose[:3],
                "accepted_evidence": dict(command.get("accepted_evidence") or {}),
                "same_pose_flip_count": max(
                    0, int(command.get("same_pose_flip_count", 0) or 0)
                ),
                "requested_at": time.monotonic(),
            }
        # ``move_base.cancel_goal`` in the approach thread is asynchronous.  In
        # particular, its final DWA command can still be applied for a few
        # simulator steps after the state machine has entered the targeted-M1
        # barrier.  For a two-stage container, that changes the view which M1
        # is meant to judge and invalidates the outer-staging evidence.  Keep
        # issuing a zero command at this already-validated safe pose until this
        # exact request resolves; the normal response path is the only one that
        # releases the hold into an inner action or another outer viewpoint.
        self._begin_container_staging_observation_hold(
            decision_id, request_id
        )
        self.attribute_refresh_request_pub.publish(
            String(data=json.dumps(request, ensure_ascii=False, separators=(",", ":")))
        )
        rospy.loginfo(
            "[semantic_behavior_executor] requested fresh M1 interaction view "
            "target=%s after_capture_step=%d attempt=%s",
            object_id,
            minimum_capture_step,
            command.get("attempt", ""),
        )

    @staticmethod
    def _is_container_staging_observation_hold_candidate(candidate: dict | None) -> bool:
        """Whether a targeted M1 request must freeze a safe outer container pose."""

        metadata = (candidate or {}).get("metadata") or {}
        return bool(
            metadata.get("container_two_stage_approach", False)
            and str(metadata.get("container_two_stage_phase") or "staging").casefold()
            == "staging"
            and metadata.get("m1_observation_staging_required", False)
        )

    def _container_staging_observation_hold_is_current_locked(
        self, decision_id: str, request_id: str
    ) -> bool:
        """Check that this hold still owns the active targeted M1 barrier."""

        if (
            self.selection is None
            or str(self.selection.get("decision_id") or "") != str(decision_id)
            or self.machine.state != STATE_WAITING_FOR_INTERACTION_OBSERVATION
        ):
            return False
        request = dict(
            (getattr(self, "_interaction_observation_requests", {}) or {}).get(
                str(decision_id)
            )
            or {}
        )
        if str(request.get("request_id") or "") != str(request_id):
            return False
        candidate = self.machine.candidate or self.selection
        return self._is_container_staging_observation_hold_candidate(candidate)

    def _publish_container_staging_hold_stop(self) -> None:
        """Best-effort zero velocity while an outer M1 view is pending."""

        publisher = getattr(self, "cmd_vel_pub", None)
        if publisher is None:
            return
        try:
            publisher.publish(Twist())
        except Exception as exc:  # pragma: no cover - ROS transport failure
            rospy.logwarn(
                "[semantic_behavior_executor] failed to publish container M1 "
                "staging hold stop: %s",
                exc,
            )

    def _run_container_staging_observation_hold(
        self, decision_id: str, request_id: str
    ) -> None:
        """Suppress residual DWA velocity until one exact targeted M1 reply ends it."""

        interval_s = max(
            0.01,
            float(getattr(self, "interaction_observation_poll_interval_s", 0.05)),
        )
        while not rospy.is_shutdown():
            with self.lock:
                if not self._container_staging_observation_hold_is_current_locked(
                    decision_id, request_id
                ):
                    return
            # ``cancel_goal`` is sent once before this loop.  Repeating only the
            # zero velocity avoids a cancellation storm while still overriding
            # a DWA message already in flight when the request was armed.
            self._publish_container_staging_hold_stop()
            time.sleep(interval_s)

    def _begin_container_staging_observation_hold(
        self, decision_id: str, request_id: str
    ) -> bool:
        """Cancel residual navigation and start the bounded outer-staging hold."""

        with self.lock:
            if not self._container_staging_observation_hold_is_current_locked(
                decision_id, request_id
            ):
                return False
        try:
            self.move_base.cancel_goal()
        except Exception as exc:  # pragma: no cover - ROS transport failure
            rospy.logwarn(
                "[semantic_behavior_executor] failed to cancel residual "
                "navigation for container M1 staging hold: %s",
                exc,
            )
        self._publish_container_staging_hold_stop()
        threading.Thread(
            target=self._run_container_staging_observation_hold,
            args=(str(decision_id), str(request_id)),
            daemon=True,
        ).start()
        return True

    def _publish_explore_command(
        self,
        candidate: dict,
        action: str,
        success: bool | None = None,
        detail: dict | None = None,
    ) -> None:
        if action == "reserve_frontier":
            self._last_explore_reservation_publish_at = time.monotonic()
            self._explore_reservation_publish_count += 1
        payload = {
            "command_id": self._command_id(candidate),
            "decision_id": candidate.get("decision_id", ""),
            "candidate_id": candidate.get("candidate_id", ""),
            "action": action,
            "cluster_id": (candidate.get("metadata") or {}).get(
                "cluster_id", candidate.get("target_id", "")
            ),
            "goal_xyyaw": list(candidate.get("goal_xyyaw") or []),
            "frontier_point": list(
                (candidate.get("metadata") or {}).get("frontier_point") or []
            ),
            "candidate_sequence": int(candidate.get("candidate_sequence", 0) or 0),
            "graph_revision": int(candidate.get("graph_revision", 0) or 0),
        }
        if success is not None:
            payload["success"] = bool(success)
        if detail:
            payload["detail"] = dict(detail)
        self.explore_command_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )

    @staticmethod
    def _has_valid_drawer_visual_contract(interaction: dict) -> bool:
        regions = interaction.get("open_regions")
        if not isinstance(regions, list) or not regions:
            return False
        valid_region = False
        for region in regions:
            if not isinstance(region, dict):
                continue
            center = region.get("center")
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                continue
            try:
                x, y = float(center[0]), float(center[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                valid_region = True
                break
        if not valid_region:
            return False
        if SemanticBehaviorExecutor._finite_public_bbox(
            interaction.get("drawer_container_bbox_2d")
        ) is None:
            return False
        return SemanticBehaviorExecutor._public_step_or_none(
            interaction.get("drawer_container_capture_step")
        ) is not None

    def _reject_drawer_command_without_visual_contract(
        self, candidate: dict, reason: str
    ) -> None:
        """Fail locally before an empty/ungrounded drawer command reaches bridge."""

        commands: list[dict] = []
        with self.lock:
            if (
                self.selection is None
                or str(self.selection.get("decision_id") or "")
                != str(candidate.get("decision_id") or "")
                or self.machine.state != STATE_INTERACTING
            ):
                return
            commands = self.machine.on_interaction_result(
                False,
                detail={
                    "reason": reason,
                    "failure_stage": "interaction_visual_precondition",
                    "terminal_candidate_exclusion": True,
                    "action_executed": False,
                },
            )
        self._dispatch(commands)

    def _publish_interaction_command(self, candidate: dict) -> None:
        interaction = candidate.get("interaction_command") or {}
        metadata = candidate.get("metadata") or {}
        drawer_sequence_type = str(
            interaction.get("sequence_type") or ""
        ).casefold()
        if drawer_sequence_type in {"drawer_scan", "drawer_open"}:
            if not self._has_valid_drawer_visual_contract(interaction):
                self._reject_drawer_command_without_visual_contract(
                    candidate,
                    "drawer_visual_contract_missing_before_bridge",
                )
                return
        action = action_for_opaque_open_contract(
            interaction.get("action", "open"),
            enabled=self.evaluator_opaque_open_only,
        )
        with self.lock:
            self.interaction_command_sequence += 1
            interaction_sequence = self.interaction_command_sequence
        payload = {
            "command_id": self._command_id(candidate),
            "decision_id": candidate.get("decision_id", ""),
            "candidate_id": candidate.get("candidate_id", ""),
            "event_id": f"{candidate.get('decision_id', 'decision')}_interaction_{interaction_sequence:03d}",
            "node_id": interaction.get("node_id", candidate.get("target_id", "")),
            "object_id": interaction.get("object_id", candidate.get("target_name", "")),
            "node_type": str(
                interaction.get("node_type") or metadata.get("node_type") or ""
            ).casefold(),
            "action": action,
            "interaction_mode": interaction.get("interaction_mode", "open_close"),
            "container_kind": str(interaction.get("container_kind") or ""),
            "expected_state": str(
                "closed"
                if action == "close"
                else "open"
                if action == "open"
                else interaction.get("expected_state") or ""
            ),
            "sequence_type": interaction.get("sequence_type", ""),
            "operation_method": interaction.get("operation_method", "unknown"),
            "open_regions": list(interaction.get("open_regions") or []),
            "approach_goal_xyyaw": list(
                interaction.get("interaction_approach_pose_xyyaw")
                or candidate.get("goal_xyyaw")
                or []
            ),
            "visual_operation_plan": dict(
                interaction.get("visual_operation_plan") or {}
            ),
            "interaction_approach_pose_xyyaw": list(
                interaction.get("interaction_approach_pose_xyyaw") or []
            ),
            "interaction_approach_axis_xy": list(
                interaction.get("interaction_approach_axis_xy") or []
            ),
            "interaction_ready_distance_m": float(
                interaction.get("interaction_ready_distance_m", 0.45) or 0.45
            ),
            "interaction_ready_yaw_tolerance_rad": float(
                interaction.get("interaction_ready_yaw_tolerance_rad", 0.55) or 0.55
            ),
        }
        # The bridge accepts only this compact public geometry contract for a
        # fixed portal.  Do not forward graph internals, source object names,
        # joints, or asset identifiers with the interaction command.
        aperture_observation = interaction.get("portal_aperture_observation")
        if payload["node_type"] == "portal" and isinstance(aperture_observation, dict):
            payload["portal_aperture_observation"] = {
                key: aperture_observation[key]
                for key in ("door_leaf", "connectivity", "confidence")
                if key in aperture_observation
            }
        if drawer_sequence_type in {"drawer_scan", "drawer_open"}:
            drawer_box = interaction.get("drawer_container_bbox_2d")
            if isinstance(drawer_box, (list, tuple)):
                payload["drawer_container_bbox_2d"] = list(drawer_box)
            drawer_capture_step = interaction.get("drawer_container_capture_step")
            if isinstance(drawer_capture_step, int) and not isinstance(
                drawer_capture_step, bool
            ):
                payload["drawer_container_capture_step"] = drawer_capture_step
        with self.lock:
            self.pre_interaction_image_sequence = self.latest_image_sequence
            if drawer_sequence_type == "drawer_scan":
                self._arm_drawer_scan_execution_wait_locked(payload)
        self.interaction_command_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )

    def _plan_and_publish_interaction(self, decision_id: str, candidate: dict) -> None:
        with self.lock:
            if not self._interaction_is_current(decision_id):
                return
            node = self._selected_graph_node_locked(candidate)
            graph_capture_step = self.latest_graph.get("capture_step")
        interaction = dict(candidate.get("interaction_command") or {})
        requested_action = str(interaction.get("action") or "open").casefold()
        if requested_action not in {"open", "scan"}:
            requested_action = "open"
        expected_target_type = infer_visual_interaction_target_type(candidate, node)
        if expected_target_type == "drawer_container":
            planned_candidate = candidate_with_direct_drawer_scan(
                candidate,
                node,
                capture_step=graph_capture_step,
            )
            if planned_candidate is not None:
                visual_plan = dict(
                    (planned_candidate.get("interaction_command") or {}).get(
                        "visual_operation_plan"
                    )
                    or {}
                )
                with self.lock:
                    if not self._interaction_is_current(decision_id):
                        return
                    self.active_skill_plan = {
                        "visual_operation_plan": visual_plan,
                        "subactions": [],
                        "max_retries": 0,
                    }
                    self.pending_skill_actions = []
                self._publish_interaction_command(planned_candidate)
                return
        with self.lock:
            if not self._interaction_is_current(decision_id):
                return
            visual_images, target_bbox, full_image_size = (
                self._visual_interaction_images_locked(node)
            )
            metadata = candidate.get("metadata") or {}
            labels = list(metadata.get("interaction_approach_pose_labels") or [])
            option_index = max(
                0,
                int(metadata.get("interaction_approach_goal_option_index", 0) or 0),
            )
            approach_pose_label = (
                str(labels[option_index])
                if option_index < len(labels)
                else ""
            )
        context = visual_interaction_planning_context(
            object_id=str(candidate.get("target_id") or ""),
            object_name=str(candidate.get("target_name") or ""),
            expected_target_type=expected_target_type,
            requested_action=requested_action,
            target_bbox_xyxy=target_bbox,
            full_image_size=full_image_size,
            approach_pose_label=approach_pose_label,
            alternate_view_count=max(0, len(labels) - option_index - 1),
        )
        response = self.mllm_client.request_json(
            role="skill_planning",
            instruction=VISUAL_INTERACTION_PLANNING_INSTRUCTION,
            context=context,
            images=visual_images,
            response_schema=build_visual_interaction_plan_response_schema(),
            timeout_s=self.skill_timeout_s,
            max_tokens=self.skill_max_output_tokens,
            metrics_context=self._model_metrics_context(
                "skill_planning", decision_id, candidate
            ),
        )
        planned_candidate = dict(candidate)
        result_source = "rule_fallback_model_error"
        plan: dict | None = None
        if response.payload is not None and not response.error:
            try:
                plan = validate_visual_interaction_plan(
                    response.payload,
                    expected_target_type=expected_target_type,
                    requested_action=requested_action,
                )
                result_source = "model"
            except ValueError:
                result_source = "rule_fallback_invalid_response"
        if expected_target_type == "other_container":
            # Containers are never opened from a guessed side view.  Missing
            # public box evidence is itself a request for another approach.
            if plan is not None and target_bbox is None:
                plan = {
                    **plan,
                    "approach_ready": False,
                    "reposition_required": True,
                    "view_state": "unknown",
                    "open_regions": [],
                    "operation_method": "unknown",
                    "reason": "target_box_unavailable_in_current_headcam",
                }
                result_source = "model_reposition_missing_target_box"
            if plan is None or not bool(plan.get("approach_ready")) or bool(
                plan.get("reposition_required")
            ):
                if plan is None and not result_source.startswith("rule_fallback"):
                    result_source = "model_reposition_required"
                elif plan is None:
                    result_source = f"{result_source}:reobserve_container"
                commands: list[dict] = []
                with self.lock:
                    self._append_model_event_locked(
                        "skill_planning",
                        response,
                        decision_id,
                        candidate,
                        result_source,
                    )
                    if self._interaction_is_current(decision_id):
                        commands = self._retry_interaction_approach_after_visual_gate_locked(
                            {
                                "reason": "visual_reposition_required",
                                "failure_reason": "visual_reposition_required",
                                "visual_plan": dict(plan or {}),
                                "model_result_source": result_source,
                                "target_bbox_available": target_bbox is not None,
                            }
                        )
                self._dispatch(commands)
                return
        if plan is not None:
            planned_candidate = candidate_with_visual_operation_plan(
                planned_candidate, plan
            )
            with self.lock:
                self.active_skill_plan = {
                    "visual_operation_plan": plan,
                    "subactions": [],
                    "max_retries": 0,
                }
                self.pending_skill_actions = []
        with self.lock:
            self._append_model_event_locked(
                "skill_planning",
                response,
                decision_id,
                candidate,
                result_source,
            )
        with self.lock:
            if not self._interaction_is_current(decision_id):
                return
        self._publish_interaction_command(planned_candidate)

    def _candidate_for_skill_action(self, candidate: dict, action: dict) -> dict:
        planned = dict(candidate)
        interaction = dict(planned.get("interaction_command") or {})
        interaction["action"] = (
            "close" if action.get("skill") == "close_part" else "open"
        )
        interaction["expected_state"] = (
            "closed" if interaction["action"] == "close" else "open"
        )
        part_id = str(action.get("part_id") or "")
        if part_id:
            interaction["region_id"] = part_id
        planned["interaction_command"] = interaction
        return planned

    def _run_visual_verification(
        self,
        decision_id: str,
        backend_payload: dict,
        result_image_sequence: int | None = None,
    ) -> None:
        required_sequence = (
            self.pre_interaction_image_sequence
            if result_image_sequence is None
            else int(result_image_sequence)
        )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not rospy.is_shutdown():
            with self.lock:
                if self.latest_image_sequence > required_sequence:
                    break
            time.sleep(0.05)
        with self.lock:
            if not self._interaction_is_current(decision_id):
                return
            selection = dict(self.selection or {})
            node = self._selected_graph_node_locked(selection)
            after = self._object_crop_data_locked(node)
            max_retries = int(self.active_skill_plan.get("max_retries", 1) or 0)
        execution_status = {
            "backend_success": bool(backend_payload.get("success")),
            "status": str(backend_payload.get("status") or ""),
            "failure_reason": str(
                backend_payload.get("failure_reason")
                or backend_payload.get("reason")
                or ""
            )[:160],
            "post_state": str(
                backend_payload.get("post_state") or backend_payload.get("state") or ""
            )[:64],
        }
        response = self.mllm_client.request_json(
            role="visual_verification",
            instruction=(
                "Inspect only the current cropped image of the interaction target after execution. "
                "Determine whether it now matches the requested state using the expected action "
                "and visible evidence only. Do not infer a before-image or an approach view. "
                "Return exactly one compact JSON object with success, "
                "confidence, reason, observed_states, new_contents_visible, and retry_action. "
                "Set observed_states.target_state to open, closed, ajar, unchanged, or unknown; "
                "set observed_states.visible_change to yes, no, or unknown; and set retry_action "
                "to none, retry, reposition, or rescan. Use a reason no longer than twelve words; "
                "do not output markdown or extra fields."
            ),
            context={
                "target": {
                    "id": "target",
                    "expected_action": str(
                        (selection.get("interaction_command") or {}).get("action")
                        or "open"
                    ),
                    "expected_state": str(
                        (selection.get("interaction_command") or {}).get("expected_state")
                        or "open"
                    ),
                    "interaction_part": str(
                        (selection.get("interaction_command") or {}).get("region_id")
                        or ""
                    ),
                },
                "execution": execution_status,
            },
            images=[after] if after else [],
            response_schema=build_visual_verification_response_schema(),
            timeout_s=self.verification_timeout_s,
            max_tokens=self.verification_max_output_tokens,
            metrics_context=self._model_metrics_context(
                "visual_verification", decision_id, selection
            ),
        )
        commands = []
        with self.lock:
            if not self._interaction_is_current(decision_id):
                return
            if response.payload is None or response.error:
                result_source = "model_error"
                commands = self.machine.on_verification_result(
                    False,
                    detail={
                        "verification_mode": "mllm_visual_unavailable",
                        "reason": str(response.error or "empty_model_response"),
                        "execution": execution_status,
                        "model_metrics": response.metrics(),
                    },
                )
            else:
                try:
                    verification = validate_visual_verification(response.payload)
                except ValueError as exc:
                    result_source = "model_invalid_response"
                    commands = self.machine.on_verification_result(
                        False,
                        detail={
                        "verification_mode": "mllm_visual_invalid",
                        "reason": f"invalid_model_response: {exc}",
                        "execution": execution_status,
                            "model_metrics": response.metrics(),
                        },
                    )
                else:
                    result_source = "model"
                    retry = (
                        not verification["success"]
                        and verification.get("retry_action") not in {"", "none"}
                        and self.verification_retries < max_retries
                    )
                    if retry:
                        self.verification_retries += 1
                    commands = self.machine.on_verification_result(
                        bool(verification["success"]),
                        detail={
                            **verification,
                            "verification_mode": "mllm_visual",
                            "execution": execution_status,
                            "model_metrics": response.metrics(),
                        },
                        retry=retry,
                    )
            self._append_model_event_locked(
                "visual_verification",
                response,
                decision_id,
                selection,
                result_source,
            )
        self._dispatch(commands)

    def _run_visual_verification_audit(
        self,
        decision_id: str,
        backend_payload: dict,
        result_image_sequence: int,
        selection: dict,
        node: dict,
    ) -> None:
        """Audit a sealed interaction result without re-entering the FSM.

        The backend/evaluator owns execution success.  This background M3 call
        is useful for debugging visual disagreement and can request one fresh
        Module-1 observation, but it has no path back to ``interact`` and thus
        can never replay a command that the backend already accepted.
        """

        required_sequence = int(result_image_sequence or 0)
        wait_s = max(
            0.0, float(getattr(self, "post_interaction_visual_audit_wait_s", 2.0))
        )
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            with self.lock:
                if int(getattr(self, "latest_image_sequence", 0) or 0) > required_sequence:
                    break
            time.sleep(0.05)

        with self.lock:
            after = self._object_crop_data_locked(dict(node or {}))
            image_sequence = int(getattr(self, "latest_image_sequence", 0) or 0)
            capture_step = self._public_step_or_none(
                getattr(self, "_latest_rgb_step_seq", None)
            )

        execution_status = {
            "backend_success": bool(
                backend_payload.get("success")
                or str(backend_payload.get("status") or "").upper() == "SUCCEEDED"
            ),
            "status": str(backend_payload.get("status") or ""),
            "failure_reason": str(
                backend_payload.get("failure_reason")
                or backend_payload.get("reason")
                or ""
            )[:160],
            "post_state": str(
                backend_payload.get("post_state") or backend_payload.get("state") or ""
            )[:64],
        }
        result_source = "model"
        model_metrics: dict = {}
        verification: dict | None = None
        if not after:
            result_source = "post_action_image_unavailable"
        else:
            response = self.mllm_client.request_json(
                role="visual_verification",
                instruction=(
                    "Inspect only the current cropped image of the interaction target after execution. "
                    "Determine whether it now matches the requested state using visible evidence only. "
                    "This is an audit: never request an action. Return exactly one compact JSON object "
                    "with success, confidence, reason, observed_states, new_contents_visible, and retry_action. "
                    "Set observed_states.target_state to open, closed, ajar, unchanged, or unknown; "
                    "set observed_states.visible_change to yes, no, or unknown; retry_action to none, retry, "
                    "reposition, or rescan; no markdown or extra fields."
                ),
                context={
                    "target": {
                        "id": "target",
                        "expected_action": str(
                            (selection.get("interaction_command") or {}).get("action")
                            or "open"
                        ),
                        "expected_state": str(
                            (selection.get("interaction_command") or {}).get("expected_state")
                            or "open"
                        ),
                    },
                    "execution": execution_status,
                },
                images=[after],
                response_schema=build_visual_verification_response_schema(),
                timeout_s=self.verification_timeout_s,
                max_tokens=self.verification_max_output_tokens,
                metrics_context=self._model_metrics_context(
                    "visual_verification_audit", decision_id, selection
                ),
            )
            model_metrics = response.metrics()
            if response.payload is None or response.error:
                result_source = "model_error"
                verification = {
                    "success": False,
                    "reason": str(response.error or "empty_model_response")[:160],
                    "retry_action": "rescan",
                }
            else:
                try:
                    verification = validate_visual_verification(response.payload)
                except ValueError as exc:
                    result_source = "model_invalid_response"
                    verification = {
                        "success": False,
                        "reason": f"invalid_model_response: {exc}"[:160],
                        "retry_action": "rescan",
                    }

        audit_success = bool((verification or {}).get("success"))
        interaction = dict(selection.get("interaction_command") or {})
        command_id = self._command_id(selection)
        audit_event = {
            "command_id": command_id,
            "decision_id": decision_id,
            "candidate_id": str(selection.get("candidate_id") or ""),
            "target_id": str(selection.get("target_id") or ""),
            "backend_success": execution_status["backend_success"],
            "audit_success": audit_success,
            "result_source": result_source,
            "reason": str((verification or {}).get("reason") or "")[:160],
            "retry_action": str((verification or {}).get("retry_action") or "none"),
            "required_image_sequence": required_sequence,
            "observed_image_sequence": image_sequence,
            "observed_capture_step": capture_step,
            "model_metrics": model_metrics,
            "timestamp": time.time(),
        }
        refresh_request = None
        with self.lock:
            audits = getattr(self, "_post_interaction_visual_audits", None)
            if audits is None:
                audits = deque(maxlen=128)
                self._post_interaction_visual_audits = audits
            # The graph-facing M1 update is intentionally bounded per sealed
            # command.  It refreshes evidence only; no state-machine retry is
            # created from an audit disagreement.
            audit_counts = getattr(
                self, "_post_interaction_visual_audit_reobserve_counts", None
            )
            if audit_counts is None:
                audit_counts = {}
                self._post_interaction_visual_audit_reobserve_counts = audit_counts
            max_reobserves = max(
                0,
                int(
                    getattr(
                        self,
                        "post_interaction_visual_audit_max_reobserves",
                        1,
                    )
                    or 0
                ),
            )
            attempted = int(audit_counts.get(command_id, 0) or 0)
            object_id = str(
                interaction.get("object_id")
                or selection.get("target_id")
                or ""
            )
            if not audit_success and object_id and attempted < max_reobserves:
                attempted += 1
                audit_counts[command_id] = attempted
                latest_step = self._public_step_or_none(
                    getattr(self, "_latest_rgb_step_seq", None)
                )
                refresh_request = {
                    "object_id": object_id,
                    "episode_id": str(
                        selection.get("episode_id")
                        or (getattr(self, "latest_graph", {}) or {}).get("episode_id")
                        or ""
                    ),
                    "minimum_capture_step": max(0, int(latest_step or 0)),
                    "reason": "post_interaction_visual_audit_mismatch",
                    "request_id": f"{command_id}:m3audit:{attempted}"[:96],
                }
                audit_event["reobserve_requested"] = True
            else:
                audit_event["reobserve_requested"] = False
            audits.append(dict(audit_event))

        if refresh_request is not None:
            publisher = getattr(self, "attribute_refresh_request_pub", None)
            if publisher is not None:
                publisher.publish(
                    String(
                        data=json.dumps(
                            refresh_request,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                )
        rospy.loginfo(
            "[semantic_behavior_executor] M3 post-action audit command=%s "
            "backend_success=%s audit_success=%s source=%s reobserve=%s",
            command_id,
            execution_status["backend_success"],
            audit_success,
            result_source,
            bool(refresh_request),
        )

    def _model_metrics_context(
        self, role: str, decision_id: str, candidate: dict
    ) -> dict:
        return {
            "decision_id": decision_id,
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "target_id": str(candidate.get("target_id") or ""),
            "episode_id": str(candidate.get("episode_id") or ""),
            "graph_revision": int(candidate.get("graph_revision", 0) or 0),
            "role_context": role,
        }

    def _append_model_event_locked(
        self,
        role: str,
        response,
        decision_id: str,
        candidate: dict,
        result_source: str,
    ) -> None:
        self.model_events.append(
            {
                "role": role,
                "decision_id": decision_id,
                "candidate_id": str(candidate.get("candidate_id") or ""),
                "target_id": str(candidate.get("target_id") or ""),
                "result_source": result_source,
                "metrics": response.metrics(),
            }
        )

    def _selected_graph_node_locked(self, candidate: dict) -> dict:
        target_id = str(candidate.get("target_id") or "")
        return next(
            (
                dict(node)
                for node in self.latest_graph.get("nodes") or []
                if str(node.get("id") or "") == target_id
            ),
            {},
        )

    @staticmethod
    def _public_step_or_none(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            step = int(value)
        except (TypeError, ValueError):
            return None
        return step if step >= 0 else None

    def _needs_fresh_drawer_scan_locked(self) -> bool:
        """Return whether the arrived interaction must re-ground a drawer box."""

        if self.selection is None:
            return False
        if str(self.selection.get("behavior_type") or "").upper() != "INTERACT":
            return False
        if is_container_two_stage_physical_action(self.selection):
            # This drawer already owns a request-ID-matched outer M1 frame and
            # its visual action regions were carried into the inner pose.  The
            # legacy post-arrival scan would discard them and issue a forbidden
            # close-range re-grounding request, so only non-two-stage drawers
            # use that wait path.
            return False
        node = self._selected_graph_node_locked(self.selection)
        return (
            infer_visual_interaction_target_type(self.selection, node)
            == "drawer_container"
        )

    def _drawer_scan_wait_detail_locked(
        self,
        decision_id: str,
        *,
        status: str,
        reason: str = "",
        selected_capture_step: int | None = None,
    ) -> dict:
        """Build public diagnostics for the post-arrival drawer-frame wait."""

        context = dict(self._drawer_scan_wait_contexts.get(decision_id) or {})
        now = time.monotonic()
        graph = self.latest_graph or {}
        detail = {
            "drawer_scan_wait_status": str(status),
            "drawer_scan_wait_elapsed_s": max(
                0.0, now - float(context.get("started_at", now) or now)
            ),
            "drawer_scan_wait_timeout_s": self.drawer_scan_wait_timeout_s,
            "drawer_scan_wait_attempt_count": int(context.get("attempt_count", 0) or 0),
            "drawer_scan_wait_last_reason": str(
                reason or context.get("last_reason") or ""
            ),
            "drawer_scan_wait_target_id": str(context.get("target_id") or ""),
            "drawer_scan_wait_baseline_capture_step": context.get(
                "baseline_graph_capture_step"
            ),
            "drawer_scan_wait_latest_capture_step": self._public_step_or_none(
                graph.get("capture_step")
            ),
            "drawer_scan_wait_baseline_graph_revision": context.get(
                "baseline_graph_revision"
            ),
            "drawer_scan_wait_latest_graph_revision": self._public_step_or_none(
                graph.get("graph_revision")
            ),
            "drawer_scan_wait_baseline_rgb_capture_step": context.get(
                "baseline_rgb_capture_step"
            ),
            "drawer_scan_wait_latest_rgb_capture_step": self._latest_rgb_step_seq,
            "drawer_scan_wait_baseline_rgb_image_sequence": context.get(
                "baseline_rgb_image_sequence"
            ),
            "drawer_scan_wait_latest_rgb_image_sequence": self.latest_image_sequence,
        }
        if selected_capture_step is not None:
            detail["drawer_scan_wait_selected_capture_step"] = int(
                selected_capture_step
            )
        return detail

    def _wait_for_fresh_drawer_scan(self, decision_id: str, candidate: dict) -> None:
        """Wait for a post-arrival public RGB/GT frame before scanning drawers.

        The evaluator only resolves a direct drawer scan if the submitted box
        and capture step were published together on its public perception
        stream.  Navigation may have consumed tens of seconds since the
        candidate was selected, so deliberately discard that old graph box.
        """

        with self.lock:
            if (
                self.selection is None
                or str(self.selection.get("decision_id") or "") != decision_id
                or self.machine.state != STATE_WAITING_FOR_DRAWER_SCAN
            ):
                return
            graph = self.latest_graph or {}
            self._drawer_scan_wait_contexts[decision_id] = {
                "started_at": time.monotonic(),
                "deadline_at": time.monotonic() + self.drawer_scan_wait_timeout_s,
                "target_id": str(candidate.get("target_id") or ""),
                "baseline_graph_capture_step": self._public_step_or_none(
                    graph.get("capture_step")
                ),
                "baseline_graph_revision": self._public_step_or_none(
                    graph.get("graph_revision")
                ),
                "baseline_rgb_image_sequence": int(self.latest_image_sequence),
                "baseline_rgb_capture_step": self._latest_rgb_step_seq,
                "attempt_count": 0,
                "last_reason": "waiting_for_fresh_public_frame",
            }
            baseline = dict(self._drawer_scan_wait_contexts[decision_id])
        rospy.loginfo(
            "[semantic_behavior_executor] WAIT_FOR_DRAWER_SCAN target=%s "
            "after graph_capture_step=%s rgb_capture_step=%s",
            baseline["target_id"],
            baseline["baseline_graph_capture_step"],
            baseline["baseline_rgb_capture_step"],
        )

        while not rospy.is_shutdown():
            commands = []
            with self.lock:
                if (
                    self.selection is None
                    or str(self.selection.get("decision_id") or "") != decision_id
                    or self.machine.state != STATE_WAITING_FOR_DRAWER_SCAN
                ):
                    return
                context = self._drawer_scan_wait_contexts.get(decision_id)
                if context is None:
                    return
                graph = self.latest_graph or {}
                node = self._selected_graph_node_locked(candidate)
                planned_candidate, wait_reason = fresh_direct_drawer_scan_candidate(
                    candidate,
                    node,
                    graph_capture_step=graph.get("capture_step"),
                    graph_revision=graph.get("graph_revision"),
                    minimum_graph_capture_step=context.get(
                        "baseline_graph_capture_step"
                    ),
                    minimum_graph_revision=context.get("baseline_graph_revision"),
                    rgb_image_sequence=self.latest_image_sequence,
                    minimum_rgb_image_sequence=context.get(
                        "baseline_rgb_image_sequence"
                    ),
                    rgb_capture_step=self._latest_rgb_step_seq,
                    minimum_rgb_capture_step=context.get(
                        "baseline_rgb_capture_step"
                    ),
                )
                context["attempt_count"] = int(context.get("attempt_count", 0)) + 1
                context["last_reason"] = wait_reason
                if planned_candidate is not None:
                    interaction = dict(
                        planned_candidate.get("interaction_command") or {}
                    )
                    selected_capture_step = self._public_step_or_none(
                        interaction.get("drawer_container_capture_step")
                    )
                    detail = self._drawer_scan_wait_detail_locked(
                        decision_id,
                        status="ready",
                        reason=wait_reason,
                        selected_capture_step=selected_capture_step,
                    )
                    self._drawer_scan_wait_records[decision_id] = dict(detail)
                    self._drawer_scan_wait_contexts.pop(decision_id, None)
                    self.active_skill_plan = {
                        "visual_operation_plan": dict(
                            interaction.get("visual_operation_plan") or {}
                        ),
                        "subactions": [],
                        "max_retries": 0,
                    }
                    self.pending_skill_actions = []
                    commands = self.machine.on_drawer_scan_ready(
                        planned_candidate,
                        detail=detail,
                    )
                    rospy.loginfo(
                        "[semantic_behavior_executor] fresh drawer scan frame "
                        "ready target=%s capture_step=%s after %d checks",
                        candidate.get("target_id", ""),
                        selected_capture_step,
                        detail["drawer_scan_wait_attempt_count"],
                    )
                elif time.monotonic() >= float(context.get("deadline_at", 0.0)):
                    detail = self._drawer_scan_wait_detail_locked(
                        decision_id,
                        status="timeout",
                        reason=wait_reason,
                    )
                    detail["reason"] = "drawer_scan_fresh_frame_timeout"
                    self._drawer_scan_wait_records[decision_id] = dict(detail)
                    self._drawer_scan_wait_contexts.pop(decision_id, None)
                    commands = self.machine.on_drawer_scan_wait_failed(detail)
                    rospy.logwarn(
                        "[semantic_behavior_executor] drawer scan fresh-frame "
                        "timeout target=%s last_reason=%s checks=%d",
                        candidate.get("target_id", ""),
                        wait_reason,
                        detail["drawer_scan_wait_attempt_count"],
                    )
            if commands:
                self._dispatch(commands)
                return
            time.sleep(self.drawer_scan_wait_poll_interval_s)

    def _target_bbox_pixels_locked(self, node: dict) -> list[int] | None:
        """Return one clamped public target box in the current head-camera frame."""

        if self.latest_image is None:
            return None
        attributes = node.get("attributes") or {}
        box = (
            attributes.get("projected_bbox_2d")
            or node.get("projected_bbox_2d")
            or attributes.get("bbox_2d")
            or node.get("bbox_2d")
        )
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            return None
        try:
            values = [float(value) for value in box[:4]]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        height, width = self.latest_image.shape[:2]
        if width <= 1 or height <= 1:
            return None
        raw_x0, raw_y0, raw_x1, raw_y1 = [int(round(value)) for value in values]
        left, right = sorted((raw_x0, raw_x1))
        top, bottom = sorted((raw_y0, raw_y1))
        left = max(0, min(width - 1, left))
        right = max(left + 1, min(width, right))
        top = max(0, min(height - 1, top))
        bottom = max(top + 1, min(height, bottom))
        if right - left < 1 or bottom - top < 1:
            return None
        return [left, top, right, bottom]

    @staticmethod
    def _encode_jpeg_data(image: np.ndarray, max_side_px: int) -> str:
        if image is None or image.size == 0:
            return ""
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return ""
        scale = min(1.0, float(max(1, max_side_px)) / max(width, height))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")

    def _padded_target_crop_locked(self, bbox: list[int]) -> np.ndarray | None:
        if self.latest_image is None:
            return None
        left, top, right, bottom = bbox
        margin_x = int(round((right - left) * self.mllm_crop_margin_ratio))
        margin_y = int(round((bottom - top) * self.mllm_crop_margin_ratio))
        height, width = self.latest_image.shape[:2]
        left = max(0, min(width - 1, left - margin_x))
        right = min(width, max(left + 1, right + margin_x))
        top = max(0, min(height - 1, top - margin_y))
        bottom = min(height, max(top + 1, bottom + margin_y))
        crop = self.latest_image[top:bottom, left:right]
        return crop.copy() if crop.size else None

    def _object_crop_data_locked(self, node: dict) -> str:
        bbox = self._target_bbox_pixels_locked(node)
        if bbox is None:
            return ""
        crop = self._padded_target_crop_locked(bbox)
        if crop is None:
            return ""
        return self._encode_jpeg_data(
            crop,
            self.mllm_crop_max_side_px,
        )

    @staticmethod
    def _rect_overlap_area(
        left: int,
        top: int,
        right: int,
        bottom: int,
        other_left: int,
        other_top: int,
        other_right: int,
        other_bottom: int,
    ) -> int:
        return max(0, min(right, other_right) - max(left, other_left)) * max(
            0, min(bottom, other_bottom) - max(top, other_top)
        )

    def _compose_visual_interaction_evidence_locked(
        self,
        image: np.ndarray,
        bbox: list[int],
    ) -> np.ndarray:
        """Make one M3-compatible image containing full context and local detail."""

        left, top, right, bottom = bbox
        composite = image.copy()
        cv2.rectangle(
            composite,
            (left, top),
            (max(left, right - 1), max(top, bottom - 1)),
            (0, 255, 255),
            2,
        )
        cv2.putText(
            composite,
            "selected target",
            (left, max(14, top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        crop = self._padded_target_crop_locked(bbox)
        if crop is None:
            return composite
        image_height, image_width = composite.shape[:2]
        # Keep the full headcam legible. The inset consumes at most about a
        # third of the frame and is placed where it hides the least target-box
        # evidence. This satisfies image=1 servers while retaining M3's global
        # frontality and local-handle evidence.
        title_height = min(20, max(12, image_height // 14))
        border = max(2, min(5, image_width // 160))
        max_inset_width = min(
            max(0, image_width - 2 * border - 8),
            max(32, int(image_width * 0.38)),
        )
        max_inset_height = min(
            max(0, image_height - title_height - 2 * border - 8),
            max(32, int(image_height * 0.46)),
        )
        if max_inset_width < 16 or max_inset_height < 16:
            return composite
        crop_height, crop_width = crop.shape[:2]
        scale = min(
            float(max_inset_width) / max(1, crop_width),
            float(max_inset_height) / max(1, crop_height),
        )
        # Tiny distant targets need a useful close-up; do not blow an inset up
        # without bound because it would consume the full contextual frame.
        scale = max(0.25, min(4.0, scale))
        inset_width = max(1, int(round(crop_width * scale)))
        inset_height = max(1, int(round(crop_height * scale)))
        inset = cv2.resize(
            crop,
            (inset_width, inset_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        panel_width = inset_width + 2 * border
        panel_height = inset_height + title_height + 2 * border
        padding = max(4, border * 2)
        corners = (
            (padding, padding),
            (max(padding, image_width - panel_width - padding), padding),
            (padding, max(padding, image_height - panel_height - padding)),
            (
                max(padding, image_width - panel_width - padding),
                max(padding, image_height - panel_height - padding),
            ),
        )
        panel_left, panel_top = min(
            corners,
            key=lambda corner: self._rect_overlap_area(
                corner[0],
                corner[1],
                corner[0] + panel_width,
                corner[1] + panel_height,
                left,
                top,
                right,
                bottom,
            ),
        )
        panel_right = min(image_width, panel_left + panel_width)
        panel_bottom = min(image_height, panel_top + panel_height)
        cv2.rectangle(
            composite,
            (panel_left, panel_top),
            (max(panel_left, panel_right - 1), max(panel_top, panel_bottom - 1)),
            (18, 18, 18),
            -1,
        )
        cv2.rectangle(
            composite,
            (panel_left, panel_top),
            (max(panel_left, panel_right - 1), max(panel_top, panel_bottom - 1)),
            (255, 255, 255),
            border,
        )
        cv2.putText(
            composite,
            "target crop",
            (panel_left + border, panel_top + max(11, title_height - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        crop_left = panel_left + border
        crop_top = panel_top + title_height + border
        composite[
            crop_top : crop_top + inset_height,
            crop_left : crop_left + inset_width,
        ] = inset
        return composite

    def _visual_interaction_images_locked(
        self, node: dict
    ) -> tuple[list[str], list[int] | None, list[int] | None]:
        """Build one M3 image with full-view context and a padded crop inset.

        The outline is deliberately rendered on a copy of the public headcam
        image rather than exposing an object pose or private asset metadata. A
        single image is required by the deployed Qwen vLLM image=1 setting.
        """

        if self.latest_image is None:
            return [], None, None
        image = self.latest_image.copy()
        height, width = image.shape[:2]
        bbox = self._target_bbox_pixels_locked(node)
        if bbox is None:
            return [], None, [width, height]
        composite = self._compose_visual_interaction_evidence_locked(image, bbox)
        evidence = self._encode_jpeg_data(
            composite, self.mllm_full_image_max_side_px
        )
        return ([evidence] if evidence else []), bbox, [width, height]

    def _interaction_is_current(self, decision_id: str) -> bool:
        return bool(
            self.selection is not None
            and str(self.selection.get("decision_id") or "") == decision_id
            and self.machine.state in {STATE_INTERACTING, STATE_VERIFYING}
        )

    def _current_pose(self, frame_id: str) -> tuple[float, float, float] | None:
        try:
            translation, rotation = self.tf_listener.lookupTransform(
                frame_id,
                self.base_frame,
                rospy.Time(0),
            )
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None
        yaw = tf.transformations.euler_from_quaternion(rotation)[2]
        return float(translation[0]), float(translation[1]), float(yaw)

    def _has_fresh_local_plan(
        self, navigation_started_at: float, now: float | None = None
    ) -> bool:
        now = time.monotonic() if now is None else float(now)
        with self.lock:
            received_at = self._latest_local_plan_received_at
            pose_count = self._latest_local_plan_pose_count
        return bool(
            pose_count >= self.navigation_stagnation_local_plan_min_poses
            and received_at >= float(navigation_started_at)
            and now - received_at
            <= self.navigation_stagnation_local_plan_max_age_s
        )

    def _publish_rotation(self, angular_z: float) -> None:
        command = Twist()
        command.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(command)

    def _fresh_rear_local_costmap_snapshot(self) -> tuple[OccupancyGrid | None, dict]:
        """Return a recent local costmap or a fail-closed diagnostic."""

        now = time.monotonic()
        with self.lock:
            occupancy = getattr(self, "_latest_occupancy", None)
            received_at = float(getattr(self, "_latest_occupancy_received_at", 0.0))
        if occupancy is None or not getattr(occupancy, "data", None):
            return None, {
                "reason": "rear_goal_local_costmap_unavailable",
                "costmap_age_s": None,
            }
        age_s = max(0.0, now - received_at) if received_at > 0.0 else math.inf
        if age_s > float(self.rear_goal_local_costmap_max_age_s):
            return None, {
                "reason": "rear_goal_local_costmap_stale",
                "costmap_age_s": age_s,
                "costmap_max_age_s": float(self.rear_goal_local_costmap_max_age_s),
            }
        try:
            width = int(occupancy.info.width)
            height = int(occupancy.info.height)
            resolution = float(occupancy.info.resolution)
            frame_id = str(occupancy.header.frame_id or "")
            if width <= 0 or height <= 0 or resolution <= 0.0 or not frame_id:
                raise ValueError("invalid local costmap geometry")
            if len(occupancy.data) < width * height:
                raise ValueError("short local costmap data")
            origin_xy = (
                float(occupancy.info.origin.position.x),
                float(occupancy.info.origin.position.y),
            )
            origin_yaw = _costmap_origin_yaw(occupancy)
        except (AttributeError, TypeError, ValueError, IndexError):
            return None, {
                "reason": "rear_goal_local_costmap_invalid",
                "costmap_age_s": age_s,
            }
        return occupancy, {
            "reason": "fresh_local_costmap",
            "costmap_age_s": age_s,
            "costmap_frame": frame_id,
            "costmap_width": width,
            "costmap_height": height,
            "costmap_resolution_m": resolution,
            "costmap_origin_xy": list(origin_xy),
            "costmap_origin_yaw": origin_yaw,
        }

    def _acquire_rear_goal_cmd_vel_lease(
        self, decision_id: str, *, allow_idle_action_client: bool = False
    ) -> bool:
        """Cancel DWA and prove it is no longer active before direct control."""

        with self.lock:
            active_lease = str(getattr(self, "_executor_cmd_vel_lease_mode", "") or "")
        if active_lease:
            self._last_rear_goal_recovery_detail = {
                "reason": "rear_goal_cmd_vel_lease_already_owned",
                "decision_id": decision_id,
                "active_lease": active_lease,
            }
            return False
        if not self.rear_goal_prerotate_step_sync_enabled:
            self._last_rear_goal_recovery_detail = {
                "reason": "rear_goal_step_gate_disabled",
                "decision_id": decision_id,
            }
            return False
        self._clear_rear_dwa_monitor(decision_id)
        try:
            self.move_base.cancel_goal()
            self.move_base.wait_for_result(
                rospy.Duration(self.rear_goal_cmd_vel_cancel_wait_s)
            )
            state = int(self.move_base.get_state())
        except Exception as exc:
            self._last_rear_goal_recovery_detail = {
                "reason": "rear_goal_cmd_vel_lease_unconfirmed",
                "decision_id": decision_id,
                "error": str(exc),
            }
            return False
        # actionlib status values: PENDING, ACTIVE, PREEMPTING, RECALLING.
        # Any of these means move_base may still publish a command, so direct
        # executor control fails closed instead of racing the local planner.
        if state in {0, 1, 6, 7} and not (
            allow_idle_action_client and state == 0
        ):
            self._last_rear_goal_recovery_detail = {
                "reason": "rear_goal_cmd_vel_lease_busy",
                "decision_id": decision_id,
                "move_base_state": state,
            }
            return False
        with self.lock:
            self._executor_cmd_vel_lease_mode = "rear_goal"
        return True

    def _release_rear_goal_cmd_vel_lease(self) -> None:
        with self.lock:
            self._executor_cmd_vel_lease_mode = ""

    def _clear_rear_dwa_monitor(self, decision_id: str | None = None) -> None:
        with self.lock:
            monitor = getattr(self, "_rear_dwa_monitor", None)
            if monitor is None:
                return
            if decision_id is None or str(monitor.get("decision_id") or "") == str(
                decision_id or ""
            ):
                self._rear_dwa_monitor = None

    def _rear_goal_rotation_choice(
        self,
        decision_id: str,
        frame_id: str,
        heading_target_xy: tuple[float, float] | None,
    ) -> tuple[dict | None, dict]:
        """Choose a fresh-map-safe, finite CW/CCW rear turn and lock it."""

        if heading_target_xy is None:
            return None, {"reason": "rear_goal_heading_unavailable"}
        pose = self._current_pose(frame_id)
        if pose is None:
            return None, {"reason": "rear_goal_pose_unavailable"}
        target_yaw = math.atan2(
            float(heading_target_xy[1]) - float(pose[1]),
            float(heading_target_xy[0]) - float(pose[0]),
        )
        angular_error = normalize_angle(target_yaw - float(pose[2]))
        if abs(angular_error) < float(self.rear_goal_enter_angle_rad):
            return {"status": "not_rear", "target_yaw": target_yaw}, {
                "reason": "rear_goal_forward_sector",
                "angular_error_rad": angular_error,
            }
        occupancy, costmap_detail = self._fresh_rear_local_costmap_snapshot()
        if occupancy is None:
            return None, {
                **costmap_detail,
                "angular_error_rad": angular_error,
            }
        costmap_frame = str(costmap_detail["costmap_frame"])
        costmap_pose = self._current_pose(costmap_frame)
        if costmap_pose is None:
            return None, {
                "reason": "rear_goal_costmap_pose_unavailable",
                **costmap_detail,
                "angular_error_rad": angular_error,
            }
        info = occupancy.info
        choices: list[dict] = []
        rejected: list[dict] = []
        angular_step = abs(
            float(self.rear_goal_rotate_speed_rad_s)
            * float(self.rear_goal_prerotate_control_dt_s)
        )
        for sign in (-1, 1):
            arc = _rotation_arc_for_sign(float(pose[2]), target_yaw, sign)
            required_steps = (
                0
                if arc <= float(self.rear_goal_exit_angle_rad)
                else int(
                    math.ceil(
                        (arc - float(self.rear_goal_exit_angle_rad))
                        / max(1e-6, angular_step)
                    )
                )
            )
            candidate_detail = {
                "turn_sign": sign,
                "direction": "ccw" if sign > 0 else "cw",
                "arc_rad": arc,
                "required_control_steps": required_steps,
            }
            if required_steps > int(self.rear_goal_prerotate_max_control_steps):
                rejected.append({**candidate_detail, "reason": "control_budget"})
                continue
            clear = circular_costmap_rotation_sweep_is_clear(
                occupancy.data,
                int(info.width),
                int(info.height),
                float(info.resolution),
                (
                    float(info.origin.position.x),
                    float(info.origin.position.y),
                ),
                float(costmap_detail["costmap_origin_yaw"]),
                (float(costmap_pose[0]), float(costmap_pose[1])),
                arc,
                float(self.rear_goal_rotation_sweep_step_rad),
                float(self.rear_goal_robot_radius_m),
                float(self.rear_goal_safety_margin_m),
                occupied_threshold=int(self.rear_goal_costmap_occupied_threshold),
                unknown_is_blocked=bool(self.rear_goal_unknown_is_blocked),
            )
            if clear:
                choices.append(candidate_detail)
            else:
                rejected.append({**candidate_detail, "reason": "footprint_collision"})
        if not choices:
            reason = (
                "rear_goal_both_turn_sweeps_blocked"
                if any(item.get("reason") == "footprint_collision" for item in rejected)
                else "rear_goal_rotation_control_budget"
            )
            return None, {
                "reason": reason,
                "angular_error_rad": angular_error,
                "target_yaw": target_yaw,
                "rejected_turns": rejected,
                **costmap_detail,
            }
        with self.lock:
            previous_lock = dict(self._rear_goal_turn_locks.get(decision_id) or {})
        locked_sign = previous_lock.get("turn_sign")
        selected = next(
            (choice for choice in choices if choice["turn_sign"] == locked_sign),
            None,
        )
        if selected is None:
            tie_sign = int(self.rear_goal_pi_turn_sign)
            selected = sorted(
                choices,
                key=lambda choice: (
                    float(choice["arc_rad"]),
                    0 if int(choice["turn_sign"]) == tie_sign else 1,
                ),
            )[0]
        selected = {
            **selected,
            "status": "selected",
            "target_yaw": target_yaw,
            "costmap_frame": costmap_frame,
        }
        with self.lock:
            self._rear_goal_turn_locks[decision_id] = {
                "turn_sign": int(selected["turn_sign"]),
                "target_yaw": float(target_yaw),
            }
        return selected, {
            "reason": "rear_goal_turn_selected",
            "angular_error_rad": angular_error,
            "target_yaw": target_yaw,
            "selected_turn": selected["direction"],
            "selected_turn_sign": int(selected["turn_sign"]),
            "selected_arc_rad": float(selected["arc_rad"]),
            "selected_control_steps": int(selected["required_control_steps"]),
            "rejected_turns": rejected,
            **costmap_detail,
        }

    def _drive_rear_goal_reverse_step_gated(
        self,
        decision_id: str,
        costmap_frame: str,
        target_distance_m: float,
    ) -> tuple[bool, dict]:
        """Execute only a short, fresh-costmap-checked reverse action."""

        gate = self._rear_goal_prerotate_gate
        with self.lock:
            gate.reset()
        dt = float(self.rear_goal_prerotate_control_dt_s)
        speed = abs(float(self.rear_goal_reverse_speed_mps))
        max_commands = max(
            1,
            int(self.rear_goal_reverse_max_control_steps)
            + int(self.rear_goal_prerotate_delivery_retry_steps),
        )
        start = self._current_pose(costmap_frame)
        if start is None:
            return False, {"reason": "rear_goal_reverse_pose_unavailable"}
        sent = 0
        applied = 0
        missed = 0
        last_sent_at: float | None = None
        last_progress_at = time.monotonic()
        previous_signature: tuple | None = None
        stall_timeout = max(
            0.1, float(self.rear_goal_prerotate_step_sync_stall_timeout_s)
        )
        try:
            while (
                not rospy.is_shutdown()
                and self._navigation_is_current(decision_id)
            ):
                now = time.monotonic()
                with self.lock:
                    acknowledgements = gate.take_acks()
                    diagnostics = gate.diagnostics()
                for acknowledgement in acknowledgements:
                    if acknowledgement.exact_step_sync and acknowledgement.command_applied:
                        applied += 1
                    else:
                        missed += 1
                signature = (
                    tuple(diagnostics.get("pending_rgb_steps") or []),
                    tuple(diagnostics.get("pending_fresh_gate_steps") or []),
                    diagnostics.get("last_sent_step"),
                    diagnostics.get("awaiting_ack_step"),
                    diagnostics.get("last_step_sync"),
                )
                if signature != previous_signature:
                    previous_signature = signature
                    last_progress_at = now
                awaiting = diagnostics.get("awaiting_ack_step")
                if (
                    awaiting is not None
                    and last_sent_at is not None
                    and now - last_sent_at >= stall_timeout
                ):
                    return False, {
                        "reason": "rear_goal_reverse_step_sync_stall",
                        "dispatched_control_steps": sent,
                        "applied_control_steps": applied,
                    }
                if awaiting is None and now - last_progress_at >= stall_timeout:
                    return False, {
                        "reason": "rear_goal_reverse_fresh_gate_stall",
                        "dispatched_control_steps": sent,
                        "applied_control_steps": applied,
                    }
                pose = self._current_pose(costmap_frame)
                if pose is None:
                    return False, {"reason": "rear_goal_reverse_pose_lost"}
                traveled = math.hypot(
                    float(pose[0]) - float(start[0]),
                    float(pose[1]) - float(start[1]),
                )
                if traveled >= max(0.0, float(target_distance_m) - 0.5 * speed * dt):
                    return True, {
                        "reason": "rear_goal_reverse_complete",
                        "distance_m": traveled,
                        "dispatched_control_steps": sent,
                        "applied_control_steps": applied,
                    }
                occupancy, costmap_detail = self._fresh_rear_local_costmap_snapshot()
                if occupancy is None:
                    return False, {
                        **costmap_detail,
                        "reason": "rear_goal_reverse_costmap_not_fresh",
                    }
                info = occupancy.info
                remaining = max(0.0, float(target_distance_m) - traveled)
                safe = circular_costmap_linear_sweep_distance(
                    occupancy.data,
                    int(info.width),
                    int(info.height),
                    float(info.resolution),
                    (
                        float(info.origin.position.x),
                        float(info.origin.position.y),
                    ),
                    float(costmap_detail["costmap_origin_yaw"]),
                    tuple(float(value) for value in pose),
                    -1.0,
                    remaining,
                    float(self.rear_goal_robot_radius_m),
                    float(self.rear_goal_safety_margin_m),
                    occupied_threshold=int(self.rear_goal_costmap_occupied_threshold),
                    unknown_is_blocked=bool(self.rear_goal_unknown_is_blocked),
                )
                if safe < min(remaining, self.rear_goal_reverse_min_distance_m):
                    return False, {
                        "reason": "rear_goal_reverse_sweep_blocked",
                        "safe_distance_m": safe,
                        **costmap_detail,
                    }
                if sent >= max_commands:
                    return False, {
                        "reason": "rear_goal_reverse_control_step_budget",
                        "dispatched_control_steps": sent,
                        "applied_control_steps": applied,
                    }
                with self.lock:
                    step_index = gate.consume_step(now=now)
                if step_index is None:
                    time.sleep(0.01)
                    continue
                command = Twist()
                command.linear.x = -speed
                self.cmd_vel_pub.publish(command)
                sent += 1
                last_sent_at = now
                time.sleep(0.05)
        finally:
            self.cmd_vel_pub.publish(Twist())
        return False, {
            "reason": "rear_goal_reverse_preempted_or_shutdown",
            "dispatched_control_steps": sent,
            "applied_control_steps": applied,
        }

    def _rear_goal_rotation_command_safe(self) -> bool:
        """Recheck the current circular footprint before every gated turn step."""

        occupancy, costmap_detail = self._fresh_rear_local_costmap_snapshot()
        if occupancy is None:
            self._last_rear_goal_recovery_detail = dict(costmap_detail)
            return False
        costmap_frame = str(costmap_detail["costmap_frame"])
        pose = self._current_pose(costmap_frame)
        if pose is None:
            self._last_rear_goal_recovery_detail = {
                "reason": "rear_goal_costmap_pose_unavailable",
                **costmap_detail,
            }
            return False
        info = occupancy.info
        clear = circular_costmap_footprint_is_clear(
            occupancy.data,
            int(info.width),
            int(info.height),
            float(info.resolution),
            (
                float(info.origin.position.x),
                float(info.origin.position.y),
            ),
            float(costmap_detail["costmap_origin_yaw"]),
            (float(pose[0]), float(pose[1])),
            float(self.rear_goal_robot_radius_m),
            float(self.rear_goal_safety_margin_m),
            occupied_threshold=int(self.rear_goal_costmap_occupied_threshold),
            unknown_is_blocked=bool(self.rear_goal_unknown_is_blocked),
        )
        if not clear:
            self._last_rear_goal_recovery_detail = {
                "reason": "rear_goal_rotation_footprint_blocked",
                **costmap_detail,
            }
        return bool(clear)

    def _attempt_rear_goal_reverse(
        self, decision_id: str, *, allow_idle_action_client: bool = False
    ) -> tuple[bool, dict]:
        if not self.rear_goal_reverse_enabled:
            return False, {"reason": "rear_goal_reverse_disabled"}
        occupancy, costmap_detail = self._fresh_rear_local_costmap_snapshot()
        if occupancy is None:
            return False, {
                **costmap_detail,
                "reason": "rear_goal_reverse_requires_fresh_costmap",
            }
        costmap_frame = str(costmap_detail["costmap_frame"])
        pose = self._current_pose(costmap_frame)
        if pose is None:
            return False, {"reason": "rear_goal_reverse_pose_unavailable"}
        info = occupancy.info
        safe = circular_costmap_linear_sweep_distance(
            occupancy.data,
            int(info.width),
            int(info.height),
            float(info.resolution),
            (
                float(info.origin.position.x),
                float(info.origin.position.y),
            ),
            float(costmap_detail["costmap_origin_yaw"]),
            tuple(float(value) for value in pose),
            -1.0,
            float(self.rear_goal_reverse_distance_m),
            float(self.rear_goal_robot_radius_m),
            float(self.rear_goal_safety_margin_m),
            occupied_threshold=int(self.rear_goal_costmap_occupied_threshold),
            unknown_is_blocked=bool(self.rear_goal_unknown_is_blocked),
        )
        if safe < float(self.rear_goal_reverse_min_distance_m):
            return False, {
                "reason": "rear_goal_reverse_no_safe_distance",
                "safe_distance_m": safe,
                **costmap_detail,
            }
        target = min(float(self.rear_goal_reverse_distance_m), safe)
        if not self._acquire_rear_goal_cmd_vel_lease(
            decision_id, allow_idle_action_client=allow_idle_action_client
        ):
            return False, dict(self._last_rear_goal_recovery_detail)
        try:
            success, detail = self._drive_rear_goal_reverse_step_gated(
                decision_id,
                costmap_frame,
                target,
            )
        finally:
            self._release_rear_goal_cmd_vel_lease()
        return success, {
            "rear_goal_reverse": True,
            "target_distance_m": target,
            **costmap_detail,
            **detail,
        }

    def _start_rear_dwa_monitor(
        self,
        decision_id: str,
        frame_id: str,
        heading_target_xy: tuple[float, float] | None,
        start_pose: tuple[float, ...] | None,
        start_goal_distance_m: float | None,
    ) -> None:
        with self.lock:
            self._rear_dwa_monitor = {
                "decision_id": decision_id,
                "frame_id": frame_id,
                "heading_target_xy": heading_target_xy,
                "started_at": time.monotonic(),
                "start_pose": None if start_pose is None else tuple(start_pose),
                "start_goal_distance_m": start_goal_distance_m,
                "samples": deque(
                    maxlen=max(
                        2,
                        int(
                            getattr(
                                self,
                                "rear_goal_oscillation_window_steps",
                                4,
                            )
                        ),
                    )
                ),
            }

    def _rear_dwa_oscillation_detail(
        self,
        decision_id: str,
        frame_id: str,
        heading_target_xy: tuple[float, float] | None,
        pose: tuple[float, ...] | None,
        goal_distance_m: float | None,
    ) -> dict | None:
        if pose is None or heading_target_xy is None:
            return None
        with self.lock:
            monitor = self._rear_dwa_monitor
            if (
                monitor is None
                or str(monitor.get("decision_id") or "") != str(decision_id)
            ):
                return None
            samples = list(monitor.get("samples") or [])
            start_pose = monitor.get("start_pose")
            start_goal_distance_m = monitor.get("start_goal_distance_m")
        target_yaw = math.atan2(
            float(heading_target_xy[1]) - float(pose[1]),
            float(heading_target_xy[0]) - float(pose[0]),
        )
        angular_error = normalize_angle(target_yaw - float(pose[2]))
        if abs(angular_error) < float(self.rear_goal_enter_angle_rad):
            return None
        displacement = 0.0
        if start_pose is not None and len(start_pose) >= 2:
            displacement = math.hypot(
                float(pose[0]) - float(start_pose[0]),
                float(pose[1]) - float(start_pose[1]),
            )
        goal_reduction = 0.0
        if start_goal_distance_m is not None and goal_distance_m is not None:
            goal_reduction = float(start_goal_distance_m) - float(goal_distance_m)
        if not rear_dwa_oscillation_detected(
            samples,
            minimum_samples=self.rear_goal_oscillation_window_steps,
            minimum_sign_flips=self.rear_goal_oscillation_min_sign_flips,
            displacement_m=displacement,
            maximum_displacement_m=self.rear_goal_oscillation_max_displacement_m,
            goal_distance_reduction_m=goal_reduction,
            minimum_goal_distance_reduction_m=(
                self.rear_goal_oscillation_min_goal_reduction_m
            ),
        ):
            return None
        with self.lock:
            if self._rear_dwa_monitor is not None:
                self._rear_dwa_monitor["samples"].clear()
        return {
            "reason": "rear_goal_dwa_oscillation",
            "angular_error_rad": angular_error,
            "dwa_samples": samples,
            "dwa_displacement_m": displacement,
            "dwa_goal_distance_reduction_m": goal_reduction,
            "frame_id": frame_id,
        }

    def _startup_scan_is_current(self, decision_id: str) -> bool:
        with self.lock:
            return bool(
                self.selection is not None
                and str(self.selection.get("decision_id") or "") == decision_id
                and self.machine.state == STATE_SCANNING
            )

    def _startup_scan_detail(
        self,
        *,
        reason: str,
        accumulated_yaw_rad: float,
        dispatched_control_steps: int,
        acknowledged_control_steps: int,
        applied_control_steps: int,
        missed_control_steps: int,
        started_at: float,
    ) -> dict:
        with self.lock:
            gate = self._startup_scan_gate.diagnostics()
        elapsed_control_s = startup_scan_elapsed_control_s(
            acknowledged_control_steps, self.startup_scan_control_dt_s
        )
        return {
            "reason": str(reason),
            "target_yaw_rad": self.startup_scan_angle_rad,
            "accumulated_yaw_rad": float(accumulated_yaw_rad),
            "angular_speed_rad_s": self.startup_scan_angular_speed_rad_s,
            "control_dt_s": self.startup_scan_control_dt_s,
            "expected_yaw_per_control_step_rad": (
                self.startup_scan_angular_speed_rad_s * self.startup_scan_control_dt_s
            ),
            "max_control_steps": self.startup_scan_max_control_steps,
            "dispatched_control_steps": int(dispatched_control_steps),
            "acknowledged_control_steps": int(acknowledged_control_steps),
            "applied_control_steps": int(applied_control_steps),
            "missed_control_steps": int(missed_control_steps),
            "elapsed_sim_control_s": elapsed_control_s,
            "elapsed_sim_s": elapsed_control_s,
            "elapsed_wall_s": max(0.0, time.monotonic() - float(started_at)),
            "step_sync_stall_timeout_s": self.startup_scan_step_sync_stall_timeout_s,
            "step_gate": gate,
        }

    def _publish_startup_scan_progress(
        self,
        *,
        status: str,
        accumulated_yaw_rad: float,
        dispatched_control_steps: int,
        acknowledged_control_steps: int,
        applied_control_steps: int,
        missed_control_steps: int,
        started_at: float,
        reason: str = "",
    ) -> None:
        detail = self._startup_scan_detail(
            reason=reason,
            accumulated_yaw_rad=accumulated_yaw_rad,
            dispatched_control_steps=dispatched_control_steps,
            acknowledged_control_steps=acknowledged_control_steps,
            applied_control_steps=applied_control_steps,
            missed_control_steps=missed_control_steps,
            started_at=started_at,
        )
        detail["status"] = str(status)
        with self.lock:
            self._startup_scan_progress = detail

    def _handle_startup_scan_result(
        self, decision_id: str, success: bool, detail: dict
    ) -> None:
        with self.lock:
            if not self._startup_scan_is_current(decision_id):
                return
            commands = self.machine.on_scan_result(success, detail=detail)
        self._dispatch(commands)

    def _run_startup_scan(self, decision_id: str, _candidate: dict) -> None:
        """Perform a mandatory 360-degree observation scan by evaluator step.

        Each nonzero velocity command is emitted only for a paired RGB/fresh
        gate step.  The bridge converts it to a fixed ``w * control_dt`` pose
        increment; wall-clock delay only changes throughput, never the motion
        integrated into one simulator action.  Completion remains odometry
        based so position-controller tracking error cannot be hidden by a
        nominal command count.
        """

        started_at = time.monotonic()
        accumulated_yaw_rad = 0.0
        last_yaw: float | None = None
        dispatched_control_steps = 0
        acknowledged_control_steps = 0
        applied_control_steps = 0
        missed_control_steps = 0
        last_command_sent_at: float | None = None
        last_progress_publish_at = 0.0
        with self.lock:
            self._startup_scan_gate.reset()
            self._startup_scan_progress = {
                "status": "SCANNING",
                "target_yaw_rad": self.startup_scan_angle_rad,
                "angular_speed_rad_s": self.startup_scan_angular_speed_rad_s,
                "control_dt_s": self.startup_scan_control_dt_s,
                "max_control_steps": self.startup_scan_max_control_steps,
            }
        try:
            while (
                not rospy.is_shutdown()
                and self._startup_scan_is_current(decision_id)
            ):
                now = time.monotonic()
                pose = self._current_pose(self.map_frame)
                if pose is not None:
                    if last_yaw is not None:
                        accumulated_yaw_rad += abs(normalize_angle(pose[2] - last_yaw))
                    last_yaw = pose[2]
                with self.lock:
                    acknowledgements = self._startup_scan_gate.take_acks()
                for acknowledgement in acknowledgements:
                    acknowledged_control_steps += 1
                    if acknowledgement.exact_step_sync and acknowledgement.command_applied:
                        applied_control_steps += 1
                    else:
                        missed_control_steps += 1
                with self.lock:
                    gate_state = self._startup_scan_gate.diagnostics()
                if gate_state.get("awaiting_ack_step") is None:
                    last_command_sent_at = None
                if accumulated_yaw_rad >= self.startup_scan_angle_rad:
                    detail = self._startup_scan_detail(
                        reason="completed",
                        accumulated_yaw_rad=accumulated_yaw_rad,
                        dispatched_control_steps=dispatched_control_steps,
                        acknowledged_control_steps=acknowledged_control_steps,
                        applied_control_steps=applied_control_steps,
                        missed_control_steps=missed_control_steps,
                        started_at=started_at,
                    )
                    self._publish_startup_scan_progress(
                        status="COMPLETED", reason="completed",
                        accumulated_yaw_rad=accumulated_yaw_rad,
                        dispatched_control_steps=dispatched_control_steps,
                        acknowledged_control_steps=acknowledged_control_steps,
                        applied_control_steps=applied_control_steps,
                        missed_control_steps=missed_control_steps,
                        started_at=started_at,
                    )
                    self._handle_startup_scan_result(decision_id, True, detail)
                    return
                reason = startup_scan_timeout_reason(
                    acknowledged_control_steps=acknowledged_control_steps,
                    control_dt_s=self.startup_scan_control_dt_s,
                    timeout_s=self.startup_scan_timeout_s,
                    awaiting_ack_step=gate_state.get("awaiting_ack_step"),
                    last_command_sent_monotonic_s=last_command_sent_at,
                    now_monotonic_s=now,
                    step_sync_stall_timeout_s=(
                        self.startup_scan_step_sync_stall_timeout_s
                    ),
                )
                reached_step_cap = (
                    dispatched_control_steps >= self.startup_scan_max_control_steps
                    and gate_state.get("awaiting_ack_step") is None
                )
                if not reason and reached_step_cap:
                    reason = "scan_step_cap"
                if reason:
                    detail = self._startup_scan_detail(
                        reason=reason,
                        accumulated_yaw_rad=accumulated_yaw_rad,
                        dispatched_control_steps=dispatched_control_steps,
                        acknowledged_control_steps=acknowledged_control_steps,
                        applied_control_steps=applied_control_steps,
                        missed_control_steps=missed_control_steps,
                        started_at=started_at,
                    )
                    self._publish_startup_scan_progress(
                        status="FAILED", reason=reason,
                        accumulated_yaw_rad=accumulated_yaw_rad,
                        dispatched_control_steps=dispatched_control_steps,
                        acknowledged_control_steps=acknowledged_control_steps,
                        applied_control_steps=applied_control_steps,
                        missed_control_steps=missed_control_steps,
                        started_at=started_at,
                    )
                    self._handle_startup_scan_result(decision_id, False, detail)
                    return
                if dispatched_control_steps < self.startup_scan_max_control_steps:
                    with self.lock:
                        step_index = self._startup_scan_gate.consume_step(now=now)
                    if step_index is not None:
                        self._publish_rotation(self.startup_scan_angular_speed_rad_s)
                        dispatched_control_steps += 1
                        last_command_sent_at = time.monotonic()
                if now - last_progress_publish_at >= 0.2:
                    self._publish_startup_scan_progress(
                        status="SCANNING",
                        accumulated_yaw_rad=accumulated_yaw_rad,
                        dispatched_control_steps=dispatched_control_steps,
                        acknowledged_control_steps=acknowledged_control_steps,
                        applied_control_steps=applied_control_steps,
                        missed_control_steps=missed_control_steps,
                        started_at=started_at,
                    )
                    last_progress_publish_at = now
                time.sleep(0.005)
        finally:
            # A zero message cannot extend a simulator action: the bridge's
            # fresh-command rule rejects a stale publish on the next window.
            self._publish_rotation(0.0)

    def _rotate_to_yaw(
        self,
        decision_id: str,
        frame_id: str,
        target_yaw: float,
        tolerance_rad: float,
        speed_rad_s: float,
        timeout_s: float,
        turn_sign: int | None = None,
        max_prerotate_control_steps: int | None = None,
        step_sync_stall_timeout_s: float | None = None,
        step_command_gate: StepCommandGate | None = None,
        delivery_retry_steps: int | None = None,
        post_budget_settle_steps: int = 0,
        rotation_label: str = "pre-rotation",
        step_sync_budget_authoritative: bool = False,
        command_guard=None,
    ) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        committed_sign = turn_sign
        gated_prerotation = max_prerotate_control_steps is not None
        budget_authoritative = bool(
            gated_prerotation and step_sync_budget_authoritative
        )
        active_step_gate = (
            self._rear_goal_prerotate_gate
            if step_command_gate is None
            else step_command_gate
        )
        active_delivery_retry_steps = (
            self.rear_goal_prerotate_delivery_retry_steps
            if delivery_retry_steps is None
            else max(0, int(delivery_retry_steps))
        )
        # A fixed-dt command is acknowledged when the bridge closes its action
        # window, but the pose callback can lag that acknowledgement by a few
        # evaluator steps.  Do not declare a perfectly applied final command a
        # failure before observing that bounded causal tail.  This never emits
        # an additional non-zero command after the finite control budget.
        active_post_budget_settle_steps = max(0, int(post_budget_settle_steps))
        nonzero_commands_sent = 0
        acknowledged_control_steps = 0
        missed_control_steps = 0
        last_command_sent_at: float | None = None
        last_gate_progress_at = time.monotonic()
        previous_gate_signature: tuple | None = None
        gate_stall_timeout_s = max(
            0.1, float(step_sync_stall_timeout_s or 2.0)
        )
        delivery_attempt_budget = None
        if gated_prerotation:
            delivery_attempt_budget = max(
                1,
                int(max_prerotate_control_steps)
                + active_delivery_retry_steps,
            )
            # A bridge action window opens only after the observation and
            # readiness barrier.  Reset before collecting the next RGB/gate
            # pair; otherwise a cmd_vel published from an earlier RGB callback
            # is intentionally rejected as stale by RosBridgePolicy.
            with self.lock:
                active_step_gate.reset()

        def finish_prerotation(reason: str, success: bool) -> bool:
            if gated_prerotation:
                with self.lock:
                    gate_diagnostics = active_step_gate.diagnostics()
                rospy.loginfo(
                    "[semantic_behavior_executor] %s finished "
                    "reason=%s dispatched=%d applied=%d missed=%d gate=%s",
                    rotation_label,
                    reason,
                    nonzero_commands_sent,
                    acknowledged_control_steps,
                    missed_control_steps,
                    gate_diagnostics,
                )
            return success

        try:
            while (
                not rospy.is_shutdown()
                and self._navigation_is_current(decision_id)
                and (budget_authoritative or time.monotonic() < deadline)
            ):
                now = time.monotonic()
                if gated_prerotation:
                    with self.lock:
                        acknowledgements = active_step_gate.take_acks()
                        gate_diagnostics = active_step_gate.diagnostics()
                    for acknowledgement in acknowledgements:
                        if (
                            acknowledgement.exact_step_sync
                            and acknowledgement.command_applied
                        ):
                            acknowledged_control_steps += 1
                        else:
                            missed_control_steps += 1
                    gate_signature = (
                        tuple(gate_diagnostics.get("pending_rgb_steps") or []),
                        tuple(gate_diagnostics.get("pending_fresh_gate_steps") or []),
                        gate_diagnostics.get("last_sent_step"),
                        gate_diagnostics.get("awaiting_ack_step"),
                        gate_diagnostics.get("last_step_sync"),
                    )
                    if gate_signature != previous_gate_signature:
                        previous_gate_signature = gate_signature
                        last_gate_progress_at = now
                    awaiting_ack_step = gate_diagnostics.get("awaiting_ack_step")
                    if (
                        awaiting_ack_step is not None
                        and last_command_sent_at is not None
                        and now - last_command_sent_at >= gate_stall_timeout_s
                    ):
                        return finish_prerotation("step_sync_stall", False)
                    if (
                        awaiting_ack_step is None
                        and now - last_gate_progress_at >= gate_stall_timeout_s
                    ):
                        return finish_prerotation("fresh_command_gate_stall", False)
                pose = self._current_pose(frame_id)
                if pose is None:
                    time.sleep(0.05)
                    continue
                error = normalize_angle(float(target_yaw) - pose[2])
                if abs(error) <= max(0.0, float(tolerance_rad)):
                    return finish_prerotation("tolerance", True)
                if committed_sign is None:
                    committed_sign = committed_turn_sign(
                        error,
                        self.rear_goal_pi_tie_tolerance_rad,
                        self.rear_goal_pi_turn_sign,
                    )
                if gated_prerotation:
                    if nonzero_commands_sent >= int(delivery_attempt_budget):
                        reason = (
                            "cmd_vel_not_applied"
                            if acknowledged_control_steps == 0 and missed_control_steps
                            else "control_step_budget"
                        )
                        if reason == "cmd_vel_not_applied":
                            return finish_prerotation(reason, False)
                        if active_post_budget_settle_steps:
                            last_sent_step = gate_diagnostics.get("last_sent_step")
                            last_sync_step = gate_diagnostics.get("last_step_sync")
                            if (
                                last_sent_step is None
                                or last_sync_step is None
                                or int(last_sync_step)
                                < int(last_sent_step)
                                + active_post_budget_settle_steps
                            ):
                                # Keep accepting the causal pose stream, but
                                # never consume another RGB/fresh command pair.
                                time.sleep(0.005)
                                continue
                        return finish_prerotation(reason, False)
                    if command_guard is not None and not bool(command_guard()):
                        return finish_prerotation("safety_guard", False)
                    with self.lock:
                        command_step_index = active_step_gate.consume_step(
                            now=now
                        )
                    if command_step_index is None:
                        time.sleep(0.01)
                        continue
                    nonzero_commands_sent += 1
                    last_command_sent_at = now
                self._publish_rotation(float(committed_sign) * abs(float(speed_rad_s)))
                time.sleep(0.05)
        finally:
            self._publish_rotation(0.0)
        if budget_authoritative:
            return finish_prerotation("preempted_or_shutdown", False)
        return finish_prerotation("wall_timeout", False)

    def _prerotate_for_rear_goal(
        self,
        decision_id: str,
        frame_id: str,
        goal_x: float,
        goal_y: float,
        heading_target_xy: tuple[float, float] | None = None,
        *,
        trigger_source: str = "initial_rear_goal",
        allow_reverse: bool = True,
    ) -> bool:
        self._last_rear_goal_recovery_detail = {}
        if not self.rear_goal_prerotate_enabled:
            self._last_rear_goal_recovery_detail = {
                "reason": "rear_goal_prerotation_disabled",
                "trigger_source": trigger_source,
            }
            return True
        # A navigation pre-turn must follow the first reachable segment of the
        # global path.  When preflight did not produce a valid path (for
        # example an EXPLORE fail-open), using the final goal can turn the base
        # away from the path that move_base will eventually choose.
        heading_target_xy = navigation_prerotation_heading_target(heading_target_xy)
        if heading_target_xy is None:
            self._last_rear_goal_recovery_detail = {
                "reason": "rear_goal_heading_unavailable",
                "trigger_source": trigger_source,
            }
            return False
        choice, detail = self._rear_goal_rotation_choice(
            decision_id,
            frame_id,
            heading_target_xy,
        )
        detail = {**detail, "trigger_source": trigger_source}
        if choice is not None and choice.get("status") == "not_rear":
            self._last_rear_goal_recovery_detail = detail
            return True
        if choice is None:
            # A short reverse is allowed only when both turn sweeps were
            # explicitly rejected by the same fresh local costmap.  Missing
            # pose/map information is never converted into a blind backoff.
            if (
                allow_reverse
                and detail.get("reason") == "rear_goal_both_turn_sweeps_blocked"
            ):
                backed_off, reverse_detail = self._attempt_rear_goal_reverse(
                    decision_id
                )
                self._last_rear_goal_recovery_detail = {
                    **detail,
                    "rear_goal_reverse_detail": reverse_detail,
                    "reason": (
                        "rear_goal_reverse_complete_replan"
                        if backed_off
                        else "rear_goal_no_safe_turn_or_reverse"
                    ),
                }
                return False
            self._last_rear_goal_recovery_detail = detail
            return False
        if not self._acquire_rear_goal_cmd_vel_lease(decision_id):
            self._last_rear_goal_recovery_detail = {
                **detail,
                **self._last_rear_goal_recovery_detail,
            }
            return False
        try:
            rotated = self._rotate_to_yaw(
                decision_id,
                frame_id,
                float(choice["target_yaw"]),
                self.rear_goal_exit_angle_rad,
                self.rear_goal_rotate_speed_rad_s,
                self.rear_goal_prerotate_timeout_s,
                turn_sign=int(choice["turn_sign"]),
                max_prerotate_control_steps=int(choice["required_control_steps"]),
                step_sync_stall_timeout_s=(
                    self.rear_goal_prerotate_step_sync_stall_timeout_s
                ),
                step_command_gate=self._rear_goal_prerotate_gate,
                delivery_retry_steps=self.rear_goal_prerotate_delivery_retry_steps,
                post_budget_settle_steps=(
                    getattr(
                        self, "rear_goal_prerotate_post_budget_settle_steps", 3
                    )
                ),
                rotation_label="rear-goal-safe-turn",
                # The bridge applies exactly one fixed-dt target increment per
                # gate/ack pair; do not let slow ROS wall time alter this path.
                step_sync_budget_authoritative=True,
                command_guard=self._rear_goal_rotation_command_safe,
            )
        finally:
            self._release_rear_goal_cmd_vel_lease()
        turn_failure_detail = dict(self._last_rear_goal_recovery_detail)
        self._last_rear_goal_recovery_detail = {
            **detail,
            "reason": "rear_goal_turn_complete" if rotated else "rear_goal_turn_failed",
            **({"turn_failure_detail": turn_failure_detail} if not rotated else {}),
        }
        return bool(rotated)

    def _final_align_goal(
        self,
        decision_id: str,
        frame_id: str,
        goal_x: float,
        goal_y: float,
        goal_yaw: float,
    ) -> bool | None:
        if not self.final_align_enabled:
            return None
        pose = self._current_pose(frame_id)
        if pose is None:
            return None
        distance = math.hypot(goal_x - pose[0], goal_y - pose[1])
        if distance > self.final_align_max_distance_m:
            return None
        if abs(normalize_angle(goal_yaw - pose[2])) <= self.final_align_yaw_tolerance_rad:
            return True
        self.move_base.cancel_goal()
        self.move_base.wait_for_result(
            rospy.Duration(max(0.0, self.final_align_cancel_wait_s))
        )
        return self._rotate_to_yaw(
            decision_id,
            frame_id,
            goal_yaw,
            self.final_align_yaw_tolerance_rad,
            self.final_align_rotate_speed_rad_s,
            self.final_align_timeout_s,
        )

    def _final_align_interaction_goal(
        self,
        decision_id: str,
        frame_id: str,
        goal_x: float,
        goal_y: float,
        goal_yaw: float,
        *,
        ready_distance_m: float,
    ) -> bool | None:
        """Bounded final yaw alignment for a bridge-validated INTERACT pose.

        The controller is allowed only inside both the configured interaction
        final-align radius and the candidate's bridge-ready distance.  It is
        intentionally independent of generic ``final_align_enabled``.
        """

        if not self.interaction_final_align_enabled:
            return None
        pose = self._current_pose(frame_id)
        if pose is None:
            return None
        allowed_distance_m = min(
            self.interaction_final_align_max_distance_m,
            max(0.05, float(ready_distance_m)),
        )
        distance = math.hypot(goal_x - pose[0], goal_y - pose[1])
        if distance > allowed_distance_m:
            return None
        if (
            abs(normalize_angle(goal_yaw - pose[2]))
            <= self.interaction_final_align_yaw_tolerance_rad
        ):
            return True
        self.move_base.cancel_goal()
        self.move_base.wait_for_result(
            rospy.Duration(max(0.0, self.final_align_cancel_wait_s))
        )
        rotation_kwargs: dict = {}
        if self.interaction_final_align_step_sync_enabled:
            yaw_error = normalize_angle(goal_yaw - pose[2])
            base_control_step_budget = prerotation_control_step_budget(
                yaw_error,
                self.interaction_final_align_yaw_tolerance_rad,
                self.interaction_final_align_rotate_speed_rad_s,
                self.interaction_final_align_control_dt_s,
                self.interaction_final_align_max_control_steps,
            )
            tracking_margin_steps = max(
                0,
                int(
                    getattr(
                        self,
                        "interaction_final_align_control_step_margin_steps",
                        0,
                    )
                ),
            )
            control_step_budget = min(
                self.interaction_final_align_max_control_steps,
                base_control_step_budget + tracking_margin_steps,
            )
            if control_step_budget <= 0:
                return True
            rospy.loginfo(
                "[semantic_behavior_executor] interaction final-align "
                "yaw_error=%.3f base_budget=%d budget=%d margin=%d cap=%d "
                "dt=%.3f speed=%.3f tol=%.3f",
                yaw_error,
                base_control_step_budget,
                control_step_budget,
                tracking_margin_steps,
                self.interaction_final_align_max_control_steps,
                self.interaction_final_align_control_dt_s,
                self.interaction_final_align_rotate_speed_rad_s,
                self.interaction_final_align_yaw_tolerance_rad,
            )
            rotation_kwargs = {
                # The shared yaw controller retains this legacy parameter name,
                # but the dedicated gate makes it an interaction-final-align
                # control budget rather than a rear-goal pre-rotation budget.
                "max_prerotate_control_steps": control_step_budget,
                "step_sync_stall_timeout_s": (
                    self.interaction_final_align_step_sync_stall_timeout_s
                ),
                "step_command_gate": self._interaction_final_align_gate,
                "delivery_retry_steps": (
                    self.interaction_final_align_delivery_retry_steps
                ),
                "post_budget_settle_steps": (
                    getattr(
                        self,
                        "interaction_final_align_post_budget_settle_steps",
                        3,
                    )
                ),
                "rotation_label": "interaction-final-align",
                # Fixed-dt simulator actions, acknowledgements, the finite
                # control budget and gate-stall timeout—not ROS wall time—own
                # completion for this gated interaction turn.
                "step_sync_budget_authoritative": True,
            }
        return self._rotate_to_yaw(
            decision_id,
            frame_id,
            goal_yaw,
            self.interaction_final_align_yaw_tolerance_rad,
            self.interaction_final_align_rotate_speed_rad_s,
            self.interaction_final_align_timeout_s,
            **rotation_kwargs,
        )

    def _wait_for_next_interaction_pose_poll_step(
        self, decision_id: str, previous_step_index: int | None
    ) -> int | None:
        """Wait for one fresh evaluator step with no wall-clock deadline."""

        while (
            not rospy.is_shutdown()
            and self._navigation_is_current(decision_id)
        ):
            with self.lock:
                current_step_index = self._latest_step_sync_index
            if current_step_index is not None and current_step_index != previous_step_index:
                return int(current_step_index)
            time.sleep(0.005)
        return None

    def _poll_interaction_approach_pose(
        self,
        decision_id: str,
        candidate: dict,
        expected_pose_xyyaw: tuple[float, float, float],
        *,
        arrival_sample: dict | None = None,
    ) -> tuple[bool, dict]:
        """Poll fresh simulator-step poses, bounded by count rather than time."""

        interaction = candidate.get("interaction_command") or {}
        metadata = candidate.get("metadata") or {}
        frame_id = str(metadata.get("frame_id") or self.map_frame)
        distance_tolerance_m = float(
            interaction.get("interaction_ready_distance_m", 0.45) or 0.45
        )
        yaw_tolerance_rad = float(
            interaction.get("interaction_ready_yaw_tolerance_rad", 0.55) or 0.55
        )
        with self.lock:
            observed_step_index = self._latest_step_sync_index
        samples = []
        if isinstance(arrival_sample, dict) and bool(arrival_sample.get("valid")):
            # This sample was produced by the navigation loop against the same
            # selected outer staging pose.  It is intentionally accepted before
            # a post-cancel TF lookup, which may otherwise transiently lose the
            # just-reached pose while the evaluator advances to its next step.
            accepted_sample = dict(arrival_sample)
            samples.append(accepted_sample)
            return True, {
                "interaction_pose_validation": accepted_sample,
                "interaction_pose_poll_count": 0,
                "interaction_pose_poll_max_attempts": (
                    self.interaction_approach_pose_poll_max_attempts
                ),
                "interaction_pose_poll_samples": samples,
                "interaction_pose_poll_used_navigation_arrival": True,
            }
        for poll_index in range(self.interaction_approach_pose_poll_max_attempts):
            actual_pose = self._current_pose(frame_id)
            validation = interaction_pose_validation(
                list(expected_pose_xyyaw),
                None if actual_pose is None else list(actual_pose),
                distance_tolerance_m=distance_tolerance_m,
                yaw_tolerance_rad=yaw_tolerance_rad,
            )
            validation["poll_index"] = poll_index + 1
            validation["step_index"] = observed_step_index
            samples.append(validation)
            if validation.get("valid"):
                return True, {
                    "interaction_pose_validation": validation,
                    "interaction_pose_poll_count": poll_index + 1,
                    "interaction_pose_poll_max_attempts": (
                        self.interaction_approach_pose_poll_max_attempts
                    ),
                    "interaction_pose_poll_samples": samples,
                }
            if poll_index + 1 >= self.interaction_approach_pose_poll_max_attempts:
                break
            observed_step_index = self._wait_for_next_interaction_pose_poll_step(
                decision_id, observed_step_index
            )
            if observed_step_index is None:
                break
        return False, {
            "reason": "interaction_pose_poll_exhausted",
            "failure_reason": "interaction_pose_poll_exhausted",
            "interaction_pose_poll_count": len(samples),
            "interaction_pose_poll_max_attempts": (
                self.interaction_approach_pose_poll_max_attempts
            ),
            "interaction_pose_poll_samples": samples,
            "interaction_pose_validation": dict(samples[-1]) if samples else {},
        }

    def _complete_interaction_approach_navigation(
        self,
        decision_id: str,
        candidate: dict,
        *,
        selected_goal: tuple[float, float, float],
        selected_goal_option_index: int,
        interaction_approach_attempts: list[dict],
        goal_option_count: int,
        detail: dict,
    ) -> None:
        """Gate an INTERACT command on counted fresh-pose samples."""

        arrival_sample = self._container_safe_staging_arrival_sample(
            candidate,
            selected_goal,
            detail,
        )
        pose_ready, poll_detail = self._poll_interaction_approach_pose(
            decision_id,
            candidate,
            selected_goal,
            arrival_sample=arrival_sample,
        )
        detail = {
            **detail,
            **poll_detail,
            "effective_interaction_approach_pose_xyyaw": list(selected_goal),
            "interaction_approach_goal_option_index": int(
                selected_goal_option_index
            ),
            "interaction_approach_attempts": [
                dict(item) for item in interaction_approach_attempts
            ],
        }
        if pose_ready:
            self._handle_navigation_result(decision_id, True, detail)
            return
        if self._retry_interaction_approach(
            decision_id,
            candidate,
            selected_goal_option_index,
            interaction_approach_attempts,
            goal_option_count,
            detail,
        ):
            return
        self._handle_navigation_result(decision_id, False, detail)

    def _set_effective_interaction_approach(
        self,
        decision_id: str,
        candidate: dict,
        selected_goal: tuple[float, float, float],
        selected_goal_option_index: int,
        interaction_approach_attempts: list[dict],
    ) -> None:
        """Expose the exact move_base fallback while the approach is active."""

        with self.lock:
            if (
                self.selection is None
                or str(self.selection.get("decision_id") or "") != decision_id
                or self.machine.candidate is None
            ):
                return
            bound_candidate = candidate_with_effective_interaction_approach(
                candidate,
                list(selected_goal),
                goal_option_index=selected_goal_option_index,
                attempts=interaction_approach_attempts,
            )
            self.machine.candidate = bound_candidate
            self.selection = dict(bound_candidate)

    @staticmethod
    def _interaction_preflight_debug_attempts(
        goal_options: list[tuple[float, float, float]],
        goal_labels: list[str] | tuple[str, ...] | None,
        *,
        start_goal_option_index: int,
        attempted_goals: list[dict] | tuple[dict, ...],
        selected_goal_option_index: int | None = None,
    ) -> list[dict]:
        """Return a complete, non-control-flow preflight audit for an INTERACT.

        ``interaction_approach_attempts`` intentionally contains only goals
        which were actually sent to the navigation controller: its length is a
        retry budget.  A retry starts at a later ring index, and a successful
        preflight stops before evaluating the remaining options.  Previously
        those options disappeared from debug traces, making a wall-side
        container look as if it had fewer candidate poses than it really did.
        Keep a separate audit list with explicit skipped reasons so diagnostics
        are complete without consuming the physical retry budget.
        """

        options = list(goal_options or [])
        labels = list(goal_labels or [])
        checked_by_index: dict[int, dict] = {}
        for raw in attempted_goals or []:
            if not isinstance(raw, dict):
                continue
            try:
                index = int(raw.get("index"))
            except (TypeError, ValueError):
                continue
            checked_by_index[index] = dict(raw)
        start_index = max(0, int(start_goal_option_index))
        selected_index = (
            None
            if selected_goal_option_index is None
            else int(selected_goal_option_index)
        )
        result: list[dict] = []
        for index, goal in enumerate(options):
            if index in checked_by_index:
                item = dict(checked_by_index[index])
                item.setdefault("preflight_checked", True)
            else:
                if index < start_index:
                    reason = "skipped_prior_retry"
                elif selected_index is not None and index > selected_index:
                    reason = "skipped_after_reachable_option"
                else:
                    reason = "skipped_preflight"
                item = {
                    "index": index,
                    "goal_xyyaw": list(goal),
                    "reachable": None,
                    "preflight_reason": reason,
                    "preflight_attempts": 0,
                    "preflight_checked": False,
                    "preflight_skipped": True,
                }
            if index < len(labels):
                item.setdefault("approach_pose_label", str(labels[index]))
            result.append(item)
        return result

    @staticmethod
    def _candidate_with_interaction_preflight_debug(
        candidate: dict, debug_attempts: list[dict]
    ) -> dict:
        """Attach preflight diagnostics without changing retry accounting."""

        result = dict(candidate or {})
        metadata = dict(result.get("metadata") or {})
        metadata["interaction_approach_preflight_debug_attempts"] = [
            dict(item) for item in debug_attempts
        ]
        result["metadata"] = metadata
        return result

    def _run_navigation(
        self,
        decision_id: str,
        candidate: dict,
        start_goal_option_index: int = 0,
        interaction_approach_attempts: list[dict] | None = None,
    ) -> None:
        ready = self.move_base.wait_for_server(rospy.Duration(30.0))
        if not ready:
            self._handle_navigation_result(decision_id, False, {"reason": "move_base_unavailable"})
            return
        if not self._navigation_is_current(decision_id):
            return
        primary_goal_values = list(candidate.get("goal_xyyaw") or [])
        goal_options = navigation_goal_options(candidate)
        if not goal_options:
            self._handle_navigation_result(decision_id, False, {"reason": "missing_goal"})
            return
        behavior_type = str(candidate.get("behavior_type") or "")
        start_goal_option_index = max(0, int(start_goal_option_index))
        interaction_approach_attempts = list(interaction_approach_attempts or [])
        if start_goal_option_index >= len(goal_options):
            self._handle_navigation_result(
                decision_id,
                False,
                {
                    "reason": "interaction_approach_options_exhausted",
                    "failure_reason": "interaction_approach_options_exhausted",
                    "failure_stage": "interaction_approach_exhausted",
                    "interaction_approach_attempts": interaction_approach_attempts,
                },
            )
            return
        metadata = candidate.get("metadata") or {}
        interaction = candidate.get("interaction_command") or {}
        goal_labels = list(metadata.get("interaction_approach_pose_labels") or [])
        if behavior_type == "INTERACT":
            direct_distance_tolerance = self._interaction_navigation_pose_tolerance_m(
                candidate
            )
            direct_yaw_tolerance = float(
                interaction.get("interaction_ready_yaw_tolerance_rad", 0.55) or 0.55
            )
        else:
            direct_distance_tolerance = float(
                metadata.get("direct_goal_tolerance_m", 0.0) or 0.0
            )
            direct_yaw_tolerance = float(
                metadata.get("direct_goal_yaw_tolerance_rad", 0.0) or 0.0
            )
        if direct_distance_tolerance > 0.0:
            current_pose = self._current_pose(
                str(metadata.get("frame_id") or self.map_frame)
            )
            primary_x, primary_y, primary_yaw = goal_options[start_goal_option_index]
            if current_pose is not None:
                position_error = math.hypot(
                    primary_x - current_pose[0], primary_y - current_pose[1]
                )
                yaw_error = abs(normalize_angle(primary_yaw - current_pose[2]))
                if position_error <= direct_distance_tolerance and (
                    direct_yaw_tolerance <= 0.0 or yaw_error <= direct_yaw_tolerance
                ):
                    direct_detail = {
                        "reason": "already_at_verified_approach_pose",
                        "position_error_m": position_error,
                        "yaw_error_rad": yaw_error,
                    }
                    if str(behavior_type).upper() == "INTERACT":
                        with self.lock:
                            arrival_step_index = getattr(
                                self, "_latest_step_sync_index", None
                            )
                        direct_detail["interaction_pose_validation"] = (
                            interaction_pose_validation(
                                [primary_x, primary_y, primary_yaw],
                                list(current_pose),
                                distance_tolerance_m=direct_distance_tolerance,
                                yaw_tolerance_rad=direct_yaw_tolerance,
                            )
                        )
                        direct_detail["interaction_arrival_step_index"] = (
                            arrival_step_index
                        )
                        direct_attempts = list(interaction_approach_attempts)
                        direct_attempts.append(
                            {
                                "index": start_goal_option_index,
                                "goal_xyyaw": [primary_x, primary_y, primary_yaw],
                                "approach_pose_label": (
                                    str(goal_labels[start_goal_option_index])
                                    if start_goal_option_index < len(goal_labels)
                                    else ""
                                ),
                                "reachable": True,
                                "navigation_attempt": len(direct_attempts) + 1,
                                "outcome": "already_at_verified_approach_pose",
                            }
                        )
                        direct_debug_attempts = (
                            self._interaction_preflight_debug_attempts(
                                goal_options,
                                goal_labels,
                                start_goal_option_index=start_goal_option_index,
                                attempted_goals=[
                                    {
                                        "index": start_goal_option_index,
                                        "goal_xyyaw": [
                                            primary_x,
                                            primary_y,
                                            primary_yaw,
                                        ],
                                        "approach_pose_label": (
                                            str(goal_labels[start_goal_option_index])
                                            if start_goal_option_index < len(goal_labels)
                                            else ""
                                        ),
                                        "reachable": True,
                                        "preflight_reason": "already_at_verified_approach_pose",
                                        "preflight_attempts": 0,
                                    }
                                ],
                                selected_goal_option_index=start_goal_option_index,
                            )
                            if str(behavior_type).upper() == "INTERACT"
                            else []
                        )
                        candidate = (
                            self._candidate_with_interaction_preflight_debug(
                                candidate, direct_debug_attempts
                            )
                            if direct_debug_attempts
                            else candidate
                        )
                        self._complete_interaction_approach_navigation(
                            decision_id,
                            candidate,
                            selected_goal=(primary_x, primary_y, primary_yaw),
                            selected_goal_option_index=start_goal_option_index,
                            interaction_approach_attempts=direct_attempts,
                            goal_option_count=len(goal_options),
                            detail=direct_detail,
                        )
                    else:
                        self._handle_navigation_result(
                            decision_id,
                            True,
                            direct_detail,
                        )
                    return
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = str(
            (candidate.get("metadata") or {}).get("frame_id") or self.map_frame
        )
        goal_frame = goal.target_pose.header.frame_id
        selected_goal = None
        selected_goal_option_index = None
        path_lookahead = None
        attempted_goals = []
        is_explore = str(candidate.get("behavior_type") or "").upper() == "EXPLORE"
        is_post_interaction_traversal = (
            is_post_interaction_traversal_navigation(candidate)
        )
        post_open_costmap_detail: dict = {}
        if is_post_interaction_traversal:
            costmap_fresh, post_open_costmap_detail = (
                self._wait_for_post_interaction_costmap_freshness(
                    decision_id, candidate
                )
            )
            if not costmap_fresh:
                if (
                    str(post_open_costmap_detail.get("reason") or "")
                    == "post_open_costmap_wait_preempted"
                ):
                    return
                self._handle_navigation_result(
                    decision_id,
                    False,
                    {
                        **post_open_costmap_detail,
                        "attempted_goal_count": 0,
                        "attempted_goals": [],
                        "interaction_approach_attempts": (
                            interaction_approach_attempts
                        ),
                    },
                )
                return
        post_interaction_retry_started_at = time.monotonic()
        post_interaction_retry_deadline = (
            post_interaction_retry_started_at
            + self.post_interaction_traversal_make_plan_retry_window_s
        )
        for option_index, (option_x, option_y, option_yaw) in enumerate(
            goal_options[start_goal_option_index:], start=start_goal_option_index
        ):
            plan_reachable = False
            option_lookahead = None
            preflight_reason = ""
            actual_attempts = 0
            while True:
                actual_attempts += 1
                (
                    plan_reachable,
                    option_lookahead,
                    preflight_reason,
                ) = self._preflight_navigation_plan(
                    goal_frame, option_x, option_y, option_yaw
                )
                if is_post_interaction_traversal:
                    # Do not inherit normal-navigation's configurable
                    # make_plan_fail_open path.  This continuation is held
                    # specifically until a freshly rebuilt costmap has
                    # produced a real path through the opened portal.
                    plan_reachable = post_open_path_is_confirmed(
                        plan_reachable, preflight_reason
                    )
                retryable_post_open_path = bool(
                    is_post_interaction_traversal
                    and post_open_path_retryable_preflight_reason(preflight_reason)
                )
                if plan_reachable:
                    break
                if is_post_interaction_traversal:
                    if not retryable_post_open_path:
                        break
                elif preflight_reason != "empty_plan":
                    break
                retry_delay_s = None
                if is_post_interaction_traversal:
                    if len(goal_options) <= 1:
                        retry_delay_s = bounded_empty_plan_retry_delay(
                            time.monotonic(),
                            post_interaction_retry_deadline,
                            self.post_interaction_traversal_make_plan_retry_interval_s,
                        )
                    else:
                        # A portal continuation now carries several preserved
                        # far-side approach poses.  Preflight each one before
                        # spending the bounded retry window on the first
                        # unreachable pose; otherwise a stale/blocked option
                        # can starve every safe doorway fallback.
                        retry_delay_s = None
                elif is_explore and actual_attempts <= self.make_plan_empty_retry_count:
                    retry_delay_s = self.make_plan_empty_retry_delay_s
                if retry_delay_s is None:
                    break
                if is_post_interaction_traversal and actual_attempts == 1:
                    rospy.loginfo(
                        "[semantic_behavior_executor] retrying post-open "
                        "make_plan for at most %.2fs after a fresh global costmap",
                        self.post_interaction_traversal_make_plan_retry_window_s,
                    )
                time.sleep(retry_delay_s)
                if not self._navigation_is_current(decision_id):
                    return
                if (
                    is_post_interaction_traversal
                    and time.monotonic() >= post_interaction_retry_deadline
                ):
                    break
            if (
                is_post_interaction_traversal
                and plan_reachable
                and actual_attempts > 1
            ):
                rospy.loginfo(
                    "[semantic_behavior_executor] post-interaction traversal plan "
                    "became reachable after %d attempts (%.2fs)",
                    actual_attempts,
                    time.monotonic() - post_interaction_retry_started_at,
                )
            fail_open_empty_plan = bool(
                is_explore
                and not plan_reachable
                and preflight_reason == "empty_plan"
                and self.explore_make_plan_fail_open_after_retries
                and bool((candidate.get("metadata") or {}).get(
                    "hard_constraints_passed", True
                ))
            )
            attempted_goals.append(
                {
                    "index": option_index,
                    "goal_xyyaw": [option_x, option_y, option_yaw],
                    "approach_pose_label": (
                        str(goal_labels[option_index])
                        if option_index < len(goal_labels)
                        else ""
                    ),
                    "reachable": bool(plan_reachable or fail_open_empty_plan),
                    "preflight_reason": preflight_reason,
                    "preflight_attempts": actual_attempts,
                    "fail_open_after_empty_plan": fail_open_empty_plan,
                    "post_interaction_traversal_retry": (
                        is_post_interaction_traversal
                    ),
                }
            )
            if plan_reachable or fail_open_empty_plan:
                selected_goal = option_x, option_y, option_yaw
                selected_goal_option_index = option_index
                path_lookahead = option_lookahead
                break
        preflight_debug_attempts = (
            self._interaction_preflight_debug_attempts(
                goal_options,
                goal_labels,
                start_goal_option_index=start_goal_option_index,
                attempted_goals=attempted_goals,
                selected_goal_option_index=selected_goal_option_index,
            )
            if str(behavior_type).upper() == "INTERACT"
            else []
        )
        # An ExplorePy request is only acted on after this executor has made
        # its own preflight decision.  If the plan has become reachable, the
        # request is acknowledged as unnecessary; otherwise the sole executor
        # runs one existing costmap-gated reverse/replan attempt.
        preflight_failed = bool(
            selected_goal is None
            or (
                attempted_goals
                and bool(attempted_goals[-1].get("fail_open_after_empty_plan"))
            )
        )
        if is_explore and self._consume_external_recovery_if_needed(
            decision_id,
            candidate,
            preflight_failed=preflight_failed,
            start_goal_option_index=start_goal_option_index,
            interaction_approach_attempts=interaction_approach_attempts,
        ):
            return
        if selected_goal is None:
            post_open_wait_elapsed_s = max(
                0.0, time.monotonic() - post_interaction_retry_started_at
            )
            post_open_path_timed_out = bool(
                is_post_interaction_traversal
                and any(
                    post_open_path_retryable_preflight_reason(
                        str(item.get("preflight_reason") or "")
                    )
                    for item in attempted_goals
                )
                and time.monotonic() >= post_interaction_retry_deadline
            )
            failure_detail = {
                "reason": (
                    "post_open_path_timeout"
                    if post_open_path_timed_out
                    else "make_plan_unreachable"
                ),
                "attempted_goal_count": len(attempted_goals),
                "attempted_goals": attempted_goals,
                "interaction_approach_attempts": interaction_approach_attempts,
            }
            if preflight_debug_attempts:
                failure_detail["interaction_approach_preflight_debug_attempts"] = (
                    preflight_debug_attempts
                )
            if (
                str(behavior_type).upper() == "INTERACT"
                and is_container_two_stage_physical_action(candidate)
            ):
                # The inner physical point may be blocked even though its outer
                # M1 staging stance was reachable.  Do not retry M1 at this
                # close pose or turn it into a terminal object failure: record
                # the failed inner preflight and return to the next outer ring.
                # ``start_goal_option_index`` can be a tangent retry rather
                # than zero.  Preserve the actual last attempted inner option
                # so the retry helper advances monotonically through the
                # same-face sequence instead of restarting at tangent #1.
                failed_inner_index = max(0, int(start_goal_option_index))
                failed_goal_values: list[float] = list(
                    candidate.get("goal_xyyaw") or []
                )
                failed_label = "physical_action"
                if attempted_goals:
                    last_attempt = dict(attempted_goals[-1])
                    try:
                        failed_inner_index = max(
                            0, int(last_attempt.get("index", failed_inner_index))
                        )
                    except (TypeError, ValueError):
                        pass
                    raw_goal = list(last_attempt.get("goal_xyyaw") or [])
                    if raw_goal:
                        failed_goal_values = raw_goal
                    failed_label = str(
                        last_attempt.get("approach_pose_label") or failed_label
                    )
                elif failed_inner_index < len(goal_options):
                    failed_goal_values = list(goal_options[failed_inner_index])
                    if failed_inner_index < len(goal_labels):
                        failed_label = str(goal_labels[failed_inner_index])
                action_attempts = list(interaction_approach_attempts)
                action_attempts.append(
                    {
                        "index": failed_inner_index,
                        "goal_xyyaw": failed_goal_values,
                        "approach_pose_label": failed_label,
                        "reachable": False,
                        "outcome": str(failure_detail.get("reason") or "failed"),
                        "phase": "physical_action",
                    }
                )
                if self._retry_interaction_approach(
                    decision_id,
                    candidate,
                    failed_inner_index,
                    action_attempts,
                    len(goal_options),
                    failure_detail,
                ):
                    return
            if is_post_interaction_traversal:
                with self.lock:
                    latest_graph_revision = int(
                        self.latest_graph.get("graph_revision", 0) or 0
                    )
                    latest_graph_capture_step = self.latest_graph.get("capture_step")
                last_preflight_reason = str(
                    attempted_goals[-1].get("preflight_reason") or ""
                ) if attempted_goals else ""
                failure_detail.update(
                    {
                        "post_open_costmap": dict(post_open_costmap_detail),
                        "opened_portal_id": str(
                            metadata.get("opened_portal_id")
                            or candidate.get("target_id")
                            or ""
                        ),
                        "post_open_path_wait_elapsed_s": post_open_wait_elapsed_s,
                        "post_open_path_retry_window_s": (
                            self.post_interaction_traversal_make_plan_retry_window_s
                        ),
                        "post_open_path_timed_out": post_open_path_timed_out,
                        "post_open_path_last_preflight_reason": (
                            last_preflight_reason
                        ),
                        "post_open_path_attempt_count": sum(
                            int(item.get("preflight_attempts", 0) or 0)
                            for item in attempted_goals
                        ),
                        "post_open_path_candidate_graph_revision": int(
                            candidate.get("graph_revision", 0) or 0
                        ),
                        "post_open_path_latest_graph_revision": (
                            latest_graph_revision
                        ),
                        "post_open_path_latest_capture_step": (
                            latest_graph_capture_step
                        ),
                    }
                )
                for trace_key in (
                    "baseline_global_costmap_seq",
                    "observed_global_costmap_seq",
                    "baseline_global_costmap_header_seq",
                    "observed_global_costmap_header_seq",
                    "baseline_global_costmap_full_seq",
                    "observed_global_costmap_full_seq",
                    "baseline_global_costmap_full_header_seq",
                    "observed_global_costmap_full_header_seq",
                    "costmap_wait_elapsed",
                    "costmap_fresh",
                    "fresh_source",
                ):
                    if trace_key in post_open_costmap_detail:
                        failure_detail[trace_key] = post_open_costmap_detail[trace_key]
                if post_open_path_timed_out:
                    rospy.logwarn(
                        "[semantic_behavior_executor] post-open path timeout for "
                        "portal %s after %.2fs (%d preflight attempts)",
                        failure_detail["opened_portal_id"],
                        post_open_wait_elapsed_s,
                        sum(
                            int(item.get("preflight_attempts", 0) or 0)
                            for item in attempted_goals
                        ),
                    )
            self._handle_navigation_result(
                decision_id,
                False,
                failure_detail,
            )
            return
        x, y, yaw = selected_goal
        interaction_approach_attempt_history = list(interaction_approach_attempts)
        if str(behavior_type).upper() == "INTERACT":
            candidate = self._candidate_with_interaction_preflight_debug(
                candidate, preflight_debug_attempts
            )
            selected_attempt = dict(attempted_goals[-1])
            selected_attempt["navigation_attempt"] = (
                len(interaction_approach_attempt_history) + 1
            )
            interaction_approach_attempt_history.append(selected_attempt)
            self._set_effective_interaction_approach(
                decision_id,
                candidate,
                selected_goal,
                int(selected_goal_option_index or 0),
                interaction_approach_attempt_history,
            )
        if attempted_goals[-1]["index"] > 0:
            rospy.loginfo(
                "[semantic_behavior_executor] selected interaction fallback goal %d/%d",
                attempted_goals[-1]["index"] + 1,
                len(goal_options),
            )
        prerotated = True
        if navigation_should_prerotate(behavior_type):
            prerotated = self._prerotate_for_rear_goal(
                decision_id,
                goal_frame,
                x,
                y,
                heading_target_xy=path_lookahead,
            )
        if not prerotated:
            rospy.logwarn(
                "[semantic_behavior_executor] rear-goal safe recovery refused direct navigation: %s",
                self._last_rear_goal_recovery_detail,
            )
            self._handle_navigation_result(
                decision_id,
                False,
                dict(self._last_rear_goal_recovery_detail),
            )
            return
        if not self._navigation_is_current(decision_id):
            return
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.z = math.sin(0.5 * yaw)
        goal.target_pose.pose.orientation.w = math.cos(0.5 * yaw)
        self.move_base.send_goal(goal)
        # The two ROS callbacks may race by one scheduling turn: if ExplorePy
        # published its request just after the preflight branch above, claim it
        # immediately after this executor-owned goal is active, then cancel it
        # through the same exclusive lease before any direct reverse command.
        if is_explore and self._consume_external_recovery_if_needed(
            decision_id,
            candidate,
            preflight_failed=preflight_failed,
            start_goal_option_index=start_goal_option_index,
            interaction_approach_attempts=interaction_approach_attempts,
        ):
            return
        navigation_started_at = time.monotonic()
        start_pose = self._current_pose(goal_frame)
        start_goal_distance_m = (
            None
            if start_pose is None
            else math.hypot(x - start_pose[0], y - start_pose[1])
        )
        self._start_rear_dwa_monitor(
            decision_id,
            goal_frame,
            path_lookahead,
            start_pose,
            start_goal_distance_m,
        )
        progress_watchdog = NavigationProgressWatchdog(
            timeout_s=self.navigation_stagnation_timeout_s,
            min_displacement_m=self.navigation_stagnation_distance_m,
            min_yaw_change_rad=self.navigation_stagnation_yaw_rad,
            min_goal_distance_reduction_m=(
                self.navigation_stagnation_goal_distance_reduction_m
            ),
            allow_yaw_progress=False,
        )
        progress_watchdog.reset(
            start_pose,
            time.monotonic(),
            start_goal_distance_m,
        )
        navigation_timeout_s = (
            self.machine.config.interaction_navigation_timeout_s
            if str(candidate.get("behavior_type") or "") == "INTERACT"
            else self.machine.config.navigation_timeout_s
        )
        deadline = time.monotonic() + navigation_timeout_s
        near_goal_since = None
        require_final_yaw = navigation_requires_final_yaw(
            behavior_type,
            self.final_align_enabled,
            primary_goal_values,
        )
        state = int(self.move_base.get_state())
        while (
            not rospy.is_shutdown()
            and self._navigation_is_current(decision_id)
            and state not in TERMINAL_STATES
            and time.monotonic() < deadline
        ):
            time.sleep(0.10)
            if is_explore and self._consume_external_recovery_if_needed(
                decision_id,
                candidate,
                preflight_failed=preflight_failed,
                start_goal_option_index=start_goal_option_index,
                interaction_approach_attempts=interaction_approach_attempts,
            ):
                return
            state = int(self.move_base.get_state())
            pose = self._current_pose(goal_frame)
            now = time.monotonic()
            goal_distance_m = (
                None
                if pose is None
                else math.hypot(x - pose[0], y - pose[1])
            )
            local_plan_fresh = self._has_fresh_local_plan(
                navigation_started_at, now
            )
            # move_base can stop publishing a fresh local plan after it has
            # already brought the base to a valid interaction approach pose.
            # Do not let the semantic watchdog turn that safe, ready pose into
            # ``navigation_stagnation`` just because the planner receipt is
            # stale.  Validate against the *selected* fallback pose (rather
            # than the primary decision goal) using the same public distance /
            # yaw contract that gates the bridge interaction command.  This
            # keeps a wrong side or heading from being accepted.
            if str(behavior_type).upper() == "INTERACT" and pose is not None:
                interaction_pose_detail = interaction_pose_validation(
                    list((x, y, yaw)),
                    list(pose),
                    distance_tolerance_m=direct_distance_tolerance,
                    yaw_tolerance_rad=direct_yaw_tolerance,
                )
                if bool(interaction_pose_detail.get("valid")):
                    self.move_base.cancel_goal()
                    with self.lock:
                        arrival_step_index = getattr(
                            self, "_latest_step_sync_index", None
                        )
                    arrival_detail = {
                        "reason": "interaction_approach_pose_tolerance",
                        "goal_distance_m": goal_distance_m,
                        "local_plan_fresh": local_plan_fresh,
                        "interaction_approach_arrival_without_fresh_local_plan": (
                            not local_plan_fresh
                        ),
                        "interaction_pose_validation": interaction_pose_detail,
                        "interaction_arrival_step_index": arrival_step_index,
                    }
                    self._complete_interaction_approach_navigation(
                        decision_id,
                        candidate,
                        selected_goal=selected_goal,
                        selected_goal_option_index=int(
                            selected_goal_option_index or 0
                        ),
                        interaction_approach_attempts=(
                            interaction_approach_attempt_history
                        ),
                        goal_option_count=len(goal_options),
                        detail=arrival_detail,
                    )
                    return
                # ``final_align_max_distance_m`` is deliberately tight for
                # ordinary navigation (0.12 m by default), whereas the
                # bridge's public INTERACT contract permits a safe standoff
                # such as 0.45 m.  Once this *selected* approach pose is
                # position-ready but its heading is not, use the same bounded
                # final-align controller, capped by its interaction-specific
                # configured radius and the bridge-ready standoff.
                # Do not relax yaw tolerance or issue an action here: after
                # the turn, _complete_interaction_approach_navigation still
                # rechecks a fresh simulator-step pose before the bridge sees
                # the command.
                position_ready = bool(
                    interaction_pose_detail.get("checked")
                    and float(interaction_pose_detail.get("position_error_m", math.inf))
                    <= float(interaction_pose_detail.get("distance_tolerance_m", 0.0))
                )
                yaw_needs_alignment = bool(
                    position_ready
                    and float(interaction_pose_detail.get("yaw_error_rad", 0.0))
                    > float(interaction_pose_detail.get("yaw_tolerance_rad", math.inf))
                )
                if yaw_needs_alignment and self.interaction_final_align_enabled:
                    aligned = self._final_align_interaction_goal(
                        decision_id,
                        goal_frame,
                        x,
                        y,
                        yaw,
                        ready_distance_m=float(
                            interaction_pose_detail.get(
                                "distance_tolerance_m", direct_distance_tolerance
                            )
                        ),
                    )
                    alignment_detail = {
                        "reason": "interaction_approach_final_yaw_alignment",
                        "goal_distance_m": goal_distance_m,
                        "local_plan_fresh": local_plan_fresh,
                        "interaction_pose_validation_before_final_align": (
                            interaction_pose_detail
                        ),
                    }
                    if aligned is True:
                        self._complete_interaction_approach_navigation(
                            decision_id,
                            candidate,
                            selected_goal=selected_goal,
                            selected_goal_option_index=int(
                                selected_goal_option_index or 0
                            ),
                            interaction_approach_attempts=(
                                interaction_approach_attempt_history
                            ),
                            goal_option_count=len(goal_options),
                            detail=alignment_detail,
                        )
                        return
                    if aligned is False:
                        alignment_detail["reason"] = "final_yaw_alignment_failed"
                        if self._retry_interaction_approach(
                            decision_id,
                            candidate,
                            selected_goal_option_index,
                            interaction_approach_attempt_history,
                            len(goal_options),
                            alignment_detail,
                        ):
                            return
                        self._handle_navigation_result(
                            decision_id, False, alignment_detail
                        )
                        return
            rear_oscillation = self._rear_dwa_oscillation_detail(
                decision_id,
                goal_frame,
                path_lookahead,
                pose,
                goal_distance_m,
            )
            if rear_oscillation is not None:
                rospy.logwarn(
                    "[semantic_behavior_executor] rear DWA oscillation; taking exclusive safe turn: %s",
                    rear_oscillation,
                )
                recovered = self._prerotate_for_rear_goal(
                    decision_id,
                    goal_frame,
                    x,
                    y,
                    heading_target_xy=path_lookahead,
                    trigger_source="dwa_left_right_oscillation",
                )
                if not recovered:
                    self.move_base.cancel_goal()
                    self._handle_navigation_result(
                        decision_id,
                        False,
                        {
                            **rear_oscillation,
                            **self._last_rear_goal_recovery_detail,
                            "rear_goal_recovery": True,
                        },
                    )
                    return
                if not self._navigation_is_current(decision_id):
                    return
                goal.target_pose.header.stamp = rospy.Time.now()
                self.move_base.send_goal(goal)
                navigation_started_at = time.monotonic()
                start_pose = self._current_pose(goal_frame)
                start_goal_distance_m = (
                    None
                    if start_pose is None
                    else math.hypot(x - start_pose[0], y - start_pose[1])
                )
                progress_watchdog.reset(
                    start_pose,
                    navigation_started_at,
                    start_goal_distance_m,
                )
                self._start_rear_dwa_monitor(
                    decision_id,
                    goal_frame,
                    path_lookahead,
                    start_pose,
                    start_goal_distance_m,
                )
                near_goal_since = None
                state = int(self.move_base.get_state())
                continue
            near_final_yaw_alignment = bool(
                pose is not None
                and require_final_yaw
                and goal_distance_m is not None
                and goal_distance_m <= self.final_align_max_distance_m
            )
            if not near_final_yaw_alignment and progress_watchdog.observe(
                pose,
                now,
                goal_distance_m=goal_distance_m,
                local_plan_fresh=local_plan_fresh,
            ):
                self.move_base.cancel_goal()
                stagnation_detail = {
                    "reason": "navigation_stagnation",
                    "stagnation_timeout_s": self.navigation_stagnation_timeout_s,
                    "stagnation_distance_m": self.navigation_stagnation_distance_m,
                    "stagnation_yaw_rad": self.navigation_stagnation_yaw_rad,
                    "stagnation_goal_distance_reduction_m": (
                        self.navigation_stagnation_goal_distance_reduction_m
                    ),
                    "goal_distance_m": goal_distance_m,
                    "goal_distance_progress_m": (
                        progress_watchdog.goal_distance_reduction_m(goal_distance_m)
                    ),
                    "local_plan_fresh": local_plan_fresh,
                    "interaction_approach_attempts": (
                        interaction_approach_attempt_history
                    ),
                }
                if is_post_interaction_traversal:
                    stagnation_detail["post_open_costmap"] = dict(
                        post_open_costmap_detail
                    )
                    for trace_key in (
                        "baseline_global_costmap_seq",
                        "observed_global_costmap_seq",
                        "baseline_global_costmap_header_seq",
                        "observed_global_costmap_header_seq",
                        "baseline_global_costmap_full_seq",
                        "observed_global_costmap_full_seq",
                        "baseline_global_costmap_full_header_seq",
                        "observed_global_costmap_full_header_seq",
                        "costmap_wait_elapsed",
                        "costmap_fresh",
                        "fresh_source",
                    ):
                        if trace_key in post_open_costmap_detail:
                            stagnation_detail[trace_key] = (
                                post_open_costmap_detail[trace_key]
                            )
                if self._retry_interaction_approach(
                    decision_id,
                    candidate,
                    selected_goal_option_index,
                    interaction_approach_attempt_history,
                    len(goal_options),
                    stagnation_detail,
                ):
                    return
                self._handle_navigation_result(
                    decision_id,
                    False,
                    stagnation_detail,
                )
                return
            if require_final_yaw:
                if pose is None:
                    near_goal_since = None
                    continue
                distance = math.hypot(x - pose[0], y - pose[1])
                yaw_error = abs(normalize_angle(yaw - pose[2]))
                if (
                    distance <= self.final_align_max_distance_m
                    and yaw_error > self.final_align_yaw_tolerance_rad
                ):
                    if near_goal_since is None:
                        near_goal_since = time.monotonic()
                    elif (
                        time.monotonic() - near_goal_since
                        >= self.final_align_trigger_delay_s
                    ):
                        aligned = self._final_align_goal(
                            decision_id,
                            goal_frame,
                            x,
                            y,
                            yaw,
                        )
                        aligned_detail = {
                            "reason": "direct_final_yaw_alignment",
                            "position_error_m": distance,
                            "yaw_error_rad": yaw_error,
                        }
                        if bool(aligned) and str(behavior_type).upper() == "INTERACT":
                            self._complete_interaction_approach_navigation(
                                decision_id,
                                candidate,
                                selected_goal=selected_goal,
                                selected_goal_option_index=int(
                                    selected_goal_option_index or 0
                                ),
                                interaction_approach_attempts=(
                                    interaction_approach_attempt_history
                                ),
                                goal_option_count=len(goal_options),
                                detail=aligned_detail,
                            )
                        else:
                            self._handle_navigation_result(
                                decision_id,
                                bool(aligned),
                                aligned_detail,
                            )
                        return
                else:
                    near_goal_since = None
        if not self._navigation_is_current(decision_id):
            return
        if state not in TERMINAL_STATES:
            self.move_base.cancel_goal()
            if require_final_yaw:
                aligned = self._final_align_goal(
                    decision_id,
                    goal_frame,
                    x,
                    y,
                    yaw,
                )
                if aligned is not None:
                    timeout_alignment_detail = {
                        "reason": "navigation_timeout_final_alignment"
                    }
                    if bool(aligned) and str(behavior_type).upper() == "INTERACT":
                        self._complete_interaction_approach_navigation(
                            decision_id,
                            candidate,
                            selected_goal=selected_goal,
                            selected_goal_option_index=int(
                                selected_goal_option_index or 0
                            ),
                            interaction_approach_attempts=(
                                interaction_approach_attempt_history
                            ),
                            goal_option_count=len(goal_options),
                            detail=timeout_alignment_detail,
                        )
                    else:
                        self._handle_navigation_result(
                            decision_id,
                            bool(aligned),
                            timeout_alignment_detail,
                        )
                    return
            timeout_detail = {
                "reason": "navigation_timeout",
                "interaction_approach_attempts": interaction_approach_attempt_history,
            }
            if self._retry_interaction_approach(
                decision_id,
                candidate,
                selected_goal_option_index,
                interaction_approach_attempt_history,
                len(goal_options),
                timeout_detail,
            ):
                return
            self._handle_navigation_result(decision_id, False, timeout_detail)
            return
        success = state == GoalStatus.SUCCEEDED
        detail = {
            "status_code": state,
            "status": self.move_base.get_goal_status_text() or str(state),
        }
        if not success:
            detail["reason"] = "navigation_terminal_failure"
        if is_post_interaction_traversal:
            detail["post_open_costmap"] = dict(post_open_costmap_detail)
            for trace_key in (
                "baseline_global_costmap_seq",
                "observed_global_costmap_seq",
                "baseline_global_costmap_header_seq",
                "observed_global_costmap_header_seq",
                "baseline_global_costmap_full_seq",
                "observed_global_costmap_full_seq",
                "baseline_global_costmap_full_header_seq",
                "observed_global_costmap_full_header_seq",
                "costmap_wait_elapsed",
                "costmap_fresh",
                "fresh_source",
            ):
                if trace_key in post_open_costmap_detail:
                    detail[trace_key] = post_open_costmap_detail[trace_key]
        if str(behavior_type).upper() == "INTERACT":
            detail["interaction_approach_attempts"] = interaction_approach_attempt_history
        if success and str(behavior_type).upper() == "INTERACT":
            # A move_base terminal success may arrive between two polling-loop
            # samples.  Preserve one pose checked against this exact selected
            # approach goal so an outer container staging pose is not discarded
            # by the post-cancel TF poll.  The same public distance/yaw contract
            # still applies; this only carries the already-observed evidence
            # into _complete_interaction_approach_navigation.
            terminal_pose = self._current_pose(goal_frame)
            terminal_validation = interaction_pose_validation(
                [x, y, yaw],
                None if terminal_pose is None else list(terminal_pose),
                distance_tolerance_m=direct_distance_tolerance,
                yaw_tolerance_rad=direct_yaw_tolerance,
            )
            with self.lock:
                arrival_step_index = getattr(self, "_latest_step_sync_index", None)
            detail.update(
                {
                    "goal_distance_m": terminal_validation.get("position_error_m"),
                    "interaction_pose_validation": terminal_validation,
                    "interaction_arrival_step_index": arrival_step_index,
                    "interaction_pose_validation_source": "move_base_terminal",
                }
            )
        if success and require_final_yaw:
            aligned = self._final_align_goal(
                decision_id,
                goal_frame,
                x,
                y,
                yaw,
            )
            if aligned is False:
                success = False
                detail["reason"] = "final_yaw_alignment_failed"
            elif aligned is True:
                detail["reason"] = "final_yaw_alignment"
        if success and str(behavior_type).upper() == "INTERACT":
            self._complete_interaction_approach_navigation(
                decision_id,
                candidate,
                selected_goal=selected_goal,
                selected_goal_option_index=int(selected_goal_option_index or 0),
                interaction_approach_attempts=interaction_approach_attempt_history,
                goal_option_count=len(goal_options),
                detail=detail,
            )
            return
        if (
            not success
            and self._retry_interaction_approach(
                decision_id,
                candidate,
                selected_goal_option_index,
                interaction_approach_attempt_history,
                len(goal_options),
                detail,
            )
        ):
            return
        self._handle_navigation_result(decision_id, success, detail)

    def _interaction_approach_attempt_limit(self, candidate: dict | None) -> int:
        """Return the finite retry budget appropriate to this approach shape.

        Portal and legacy interaction candidates retain the small generic
        budget.  A two-stage container has a separately bounded outer staging
        sequence, so its physical-pose or visual failure may advance through
        that advertised sequence instead of terminating after four attempts.
        """

        metadata = (candidate or {}).get("metadata") or {}
        if bool(metadata.get("container_two_stage_approach", False)):
            staging_goals = list(
                metadata.get("container_staging_goal_xyyaw_candidates") or []
            )
            if staging_goals:
                return max(
                    1,
                    min(
                        len(staging_goals),
                        int(
                            getattr(
                                self,
                                "container_two_stage_fallback_max_attempts",
                                self.interaction_approach_fallback_max_attempts,
                            )
                        ),
                    ),
                )
        return max(1, int(self.interaction_approach_fallback_max_attempts))

    def _container_two_stage_inner_option_count(self, candidate: dict | None) -> int:
        """Return the finite same-face action sequence for the active staging face."""

        metadata = (candidate or {}).get("metadata") or {}
        if not is_container_two_stage_physical_action(candidate):
            return 0
        try:
            staging_index = max(
                0,
                int(metadata.get("container_two_stage_staging_goal_option_index", 0)),
            )
        except (TypeError, ValueError):
            return 0
        return len(
            container_two_stage_action_goal_options_for_staging(
                candidate, staging_index
            )
        )

    @staticmethod
    def _container_two_stage_outer_retry_limit(candidate: dict | None, limit: int) -> int:
        """Bound outer M1 faces independently from inner physical alternatives."""

        metadata = (candidate or {}).get("metadata") or {}
        staging_goals = list(
            metadata.get("container_staging_goal_xyyaw_candidates") or []
        )
        return max(0, min(len(staging_goals), max(0, int(limit))))

    def _retry_interaction_approach(
        self,
        decision_id: str,
        candidate: dict,
        selected_option_index: int | None,
        interaction_approach_attempts: list[dict],
        goal_option_count: int,
        failure_detail: dict,
    ) -> bool:
        attempts = [dict(attempt) for attempt in interaction_approach_attempts]
        approach_attempt_limit = self._interaction_approach_attempt_limit(candidate)
        if attempts:
            attempts[-1]["outcome"] = str(failure_detail.get("reason") or "failed")
            attempts[-1]["failure_detail"] = {
                "goal_distance_m": failure_detail.get("goal_distance_m"),
                "local_plan_fresh": failure_detail.get("local_plan_fresh"),
            }
        if is_container_two_stage_physical_action(candidate):
            metadata = candidate.get("metadata") or {}
            staging_goals = list(
                metadata.get("container_staging_goal_xyyaw_candidates") or []
            )
            try:
                completed_staging_index = max(
                    0,
                    int(
                        metadata.get(
                            "container_two_stage_staging_goal_option_index", 0
                        )
                    ),
                )
            except (TypeError, ValueError):
                return False
            # A fresh outer M1 acceptance owns a bounded sequence of physical
            # points on that *same* face.  A navigation/pose failure at one
            # point should try the remaining tangent points before forfeiting
            # the M1 evidence and asking for another outer view.
            inner_option_count = self._container_two_stage_inner_option_count(candidate)
            current_inner_index = max(0, int(selected_option_index or 0))
            next_inner_index = current_inner_index + 1
            if (
                next_inner_index < inner_option_count
                and self._navigation_is_current(decision_id)
            ):
                rospy.logwarn(
                    "[semantic_behavior_executor] inner container action approach %s; "
                    "retrying same-face physical option %d/%d without M1",
                    str(failure_detail.get("reason") or "failed"),
                    next_inner_index + 1,
                    inner_option_count,
                )
                threading.Thread(
                    target=self._run_navigation,
                    args=(decision_id, candidate, next_inner_index, attempts),
                    daemon=True,
                ).start()
                return True
            next_staging_index = completed_staging_index + 1
            outer_retry_limit = self._container_two_stage_outer_retry_limit(
                candidate, approach_attempt_limit
            )
            if (
                next_staging_index >= outer_retry_limit
                or not self._navigation_is_current(decision_id)
            ):
                return False
            rospy.logwarn(
                "[semantic_behavior_executor] inner container action approach %s; "
                "returning to outer M1 staging option %d/%d (attempt %d/%d)",
                str(failure_detail.get("reason") or "failed"),
                next_staging_index + 1,
                len(staging_goals),
                len(attempts) + 1,
                approach_attempt_limit,
            )
            with self.lock:
                if not self._navigation_is_current(decision_id):
                    return True
                commands = self.machine.retry_container_two_stage_staging(
                    next_staging_goal_option_index=next_staging_index,
                    interaction_approach_attempts=attempts,
                    detail=failure_detail,
                )
                if commands and self.machine.candidate is not None:
                    self.selection = dict(self.machine.candidate)
                accepted_by_decision = getattr(
                    self, "_container_m1_last_accepted_evidence", None
                )
                if isinstance(accepted_by_decision, dict):
                    accepted_by_decision.pop(decision_id, None)
            if not commands:
                return False
            self._dispatch(commands)
            return True
        if selected_option_index is None:
            return False
        next_option_index = next_interaction_approach_option_index(
            behavior_type=str(candidate.get("behavior_type") or ""),
            failure_detail=failure_detail,
            selected_option_index=selected_option_index,
            attempted_navigation_count=len(interaction_approach_attempts),
            max_navigation_attempts=approach_attempt_limit,
            goal_option_count=goal_option_count,
        )
        if next_option_index is None:
            return False
        if not self._navigation_is_current(decision_id):
            return True
        pose_precondition_retry = is_interaction_pose_precondition_failure(
            failure_detail
        )
        if (
            self.interaction_approach_fallback_cancel_wait_s > 0.0
            and not pose_precondition_retry
        ):
            self.move_base.wait_for_result(
                rospy.Duration(self.interaction_approach_fallback_cancel_wait_s)
            )
        if not self._navigation_is_current(decision_id):
            return True
        rospy.logwarn(
            "[semantic_behavior_executor] INTERACT approach %s; "
            "retrying option %d/%d (attempt %d/%d)",
            str(failure_detail.get("reason") or "failed"),
            next_option_index + 1,
            goal_option_count,
            len(attempts) + 1,
            approach_attempt_limit,
        )
        threading.Thread(
            target=self._run_navigation,
            args=(decision_id, candidate, next_option_index, attempts),
            daemon=True,
        ).start()
        return True

    def _preflight_navigation_plan(
        self,
        frame_id: str,
        goal_x: float,
        goal_y: float,
        goal_yaw: float,
    ) -> tuple[bool, tuple[float, float] | None, str]:
        if not self.make_plan_preflight_enabled:
            return True, None, "disabled"
        pose = self._current_pose(frame_id)
        if pose is None:
            return self.make_plan_fail_open, None, "pose_unavailable"
        stamp = rospy.Time.now()
        start = PoseStamped()
        start.header.frame_id = frame_id
        start.header.stamp = stamp
        start.pose.position.x = pose[0]
        start.pose.position.y = pose[1]
        start.pose.orientation.z = math.sin(0.5 * pose[2])
        start.pose.orientation.w = math.cos(0.5 * pose[2])
        goal = PoseStamped()
        goal.header.frame_id = frame_id
        goal.header.stamp = stamp
        goal.pose.position.x = float(goal_x)
        goal.pose.position.y = float(goal_y)
        goal.pose.orientation.z = math.sin(0.5 * float(goal_yaw))
        goal.pose.orientation.w = math.cos(0.5 * float(goal_yaw))
        try:
            rospy.wait_for_service(
                self.make_plan_service,
                timeout=max(0.0, self.make_plan_service_wait_sec),
            )
            response = self.make_plan_client(
                start=start,
                goal=goal,
                tolerance=max(0.0, self.make_plan_tolerance_m),
            )
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn_throttle(
                5.0,
                "[semantic_behavior_executor] make_plan preflight unavailable: %s",
                exc,
            )
            return self.make_plan_fail_open, None, "service_unavailable"
        poses = list(response.plan.poses or [])
        if not poses:
            return False, None, "empty_plan"
        endpoint = poses[-1].pose.position
        reachable = math.hypot(
            float(endpoint.x) - float(goal_x),
            float(endpoint.y) - float(goal_y),
        ) <= max(self.make_plan_endpoint_tolerance_m, self.make_plan_tolerance_m)
        path_xy = [
            (float(path_pose.pose.position.x), float(path_pose.pose.position.y))
            for path_pose in poses
        ]
        lookahead = path_lookahead_point(
            (pose[0], pose[1]),
            path_xy,
            self.rear_goal_lookahead_m,
        )
        return reachable, lookahead, "reachable" if reachable else "endpoint_mismatch"

    @staticmethod
    def _single_goal_recovery_failure(detail: dict) -> bool:
        """Return whether one active move_base goal deserves a safe retry.

        This deliberately excludes generic preflight failures and post-open
        traversal failures.  Those paths have their own causal-map contracts;
        this guard addresses the observed case where one ACTIVE goal reaches
        ABORTED or makes no progress before the legacy three-subgoal watchdog
        can fire.
        """

        detail = detail or {}
        reason = str(detail.get("reason") or "").strip().casefold()
        if reason == "navigation_stagnation":
            return True
        if reason != "navigation_terminal_failure":
            return False
        try:
            status_code = int(detail.get("status_code"))
        except (TypeError, ValueError):
            status_code = None
        status = str(detail.get("status") or "").strip().casefold()
        return bool(
            status_code == GoalStatus.ABORTED
            or status == "aborted"
            or "aborted" in status
        )

    def _attempt_single_goal_navigation_recovery(
        self,
        decision_id: str,
        success: bool,
        detail: dict,
    ) -> tuple[dict | None, dict]:
        """Try one executor-owned reverse/replan for an active failed goal.

        The direct command uses the same fresh local-costmap sweep, action
        lease, and step gate as rear-goal recovery.  ExplorePy remains only a
        reservation/request source in external-control mode; it never shares
        ``cmd_vel`` ownership with this path.

        Returns ``(restart, detail)``.  ``restart`` is non-None only after a
        safe reverse completed and the caller should relaunch navigation.
        ``detail`` is empty when this failure is not eligible for recovery.
        """

        if success or not bool(
            getattr(self, "navigation_failure_recovery_enabled", False)
        ):
            return None, {}
        if not self._single_goal_recovery_failure(detail):
            return None, {}
        with self.lock:
            selection = dict(self.selection or {})
            candidate = dict(self.machine.candidate or selection)
            active_state = self.machine.state
            attempts_by_decision = getattr(
                self, "_navigation_failure_recovery_attempts", {}
            )
            attempt_count = int(attempts_by_decision.get(decision_id, 0) or 0)
        if (
            not selection
            or str(selection.get("decision_id") or "") != str(decision_id)
            or active_state not in {STATE_NAVIGATING, STATE_APPROACH_INTERACTION}
            # INTERACT already owns a bounded ring of approach poses. Its
            # terminal path must remain candidate-local and deterministic;
            # adding a generic same-pose recovery after that ring is exhausted
            # would resurrect the exact loop this guard is meant to remove.
            or str(candidate.get("behavior_type") or "").upper() == "INTERACT"
            or is_post_interaction_traversal_navigation(candidate)
            or attempt_count
            >= int(getattr(self, "navigation_failure_recovery_max_attempts", 0))
        ):
            return None, {}

        with self.lock:
            attempts_by_decision = getattr(
                self, "_navigation_failure_recovery_attempts", None
            )
            if attempts_by_decision is None:
                attempts_by_decision = {}
                self._navigation_failure_recovery_attempts = attempts_by_decision
            attempts_by_decision[decision_id] = attempt_count + 1

        request = None
        is_explore = str(candidate.get("behavior_type") or "").upper() == "EXPLORE"
        if is_explore:
            request = self._take_external_recovery_request(decision_id, candidate)
            if request is not None:
                self._publish_external_recovery_feedback(
                    request,
                    status="EXECUTING",
                    accepted=True,
                    detail={"reason": "executor_owned_active_goal_recovery"},
                )

        recovered, reverse_detail = self._attempt_rear_goal_reverse(decision_id)
        recovery_detail = {
            "recovery_owner": "semantic_behavior_executor",
            "recovery_trigger": str(detail.get("reason") or ""),
            "navigation_recovery_attempt": attempt_count + 1,
            "navigation_recovery_attempt_limit": (
                int(getattr(self, "navigation_failure_recovery_max_attempts", 0))
            ),
            "reverse": dict(reverse_detail or {}),
        }
        if not recovered or not self._navigation_is_current(decision_id):
            recovery_detail["recovered"] = False
            recovery_detail["rear_goal_recovery"] = True
            if request is not None:
                self._publish_external_recovery_feedback(
                    request,
                    status="FAILED",
                    accepted=True,
                    detail=recovery_detail,
                )
            return None, recovery_detail

        metadata = candidate.get("metadata") or {}
        try:
            start_goal_option_index = max(
                0, int(metadata.get("interaction_approach_goal_option_index", 0) or 0)
            )
        except (TypeError, ValueError):
            start_goal_option_index = 0
        approach_attempts = [
            dict(item)
            for item in metadata.get("interaction_approach_attempts") or []
            if isinstance(item, dict)
        ]
        goal_options = navigation_goal_options(candidate)
        turn_attempted = False
        turn_completed = True
        if goal_options:
            selected_index = min(start_goal_option_index, len(goal_options) - 1)
            goal_x, goal_y, _goal_yaw = goal_options[selected_index]
            turn_attempted = True
            # This routine only rotates when the replanned direction is rear.
            # It rechecks the fresh circular footprint and is prevented from
            # issuing a second reverse because this recovery already moved.
            turn_completed = self._prerotate_for_rear_goal(
                decision_id,
                str(metadata.get("frame_id") or self.map_frame),
                goal_x,
                goal_y,
                heading_target_xy=(goal_x, goal_y),
                trigger_source="active_goal_recovery",
                allow_reverse=False,
            )
        recovery_detail.update(
            {
                "recovered": True,
                "turn_attempted": turn_attempted,
                "turn_completed": bool(turn_completed),
                "turn_detail": dict(self._last_rear_goal_recovery_detail),
            }
        )
        if request is not None:
            self._publish_external_recovery_feedback(
                request,
                status="SUCCEEDED",
                accepted=True,
                detail=recovery_detail,
            )
        return {
            "candidate": candidate,
            "start_goal_option_index": start_goal_option_index,
            "interaction_approach_attempts": approach_attempts,
        }, recovery_detail

    def _handle_navigation_result(
        self, decision_id: str, success: bool, detail: dict
    ) -> None:
        detail = dict(detail or {})
        self._clear_rear_dwa_monitor(decision_id)
        with self.lock:
            locks = getattr(self, "_rear_goal_turn_locks", None)
            if isinstance(locks, dict):
                locks.pop(decision_id, None)
        restart, active_goal_recovery_detail = (
            self._attempt_single_goal_navigation_recovery(
                decision_id, success, detail
            )
        )
        if restart is not None:
            rospy.logwarn(
                "[semantic_behavior_executor] recovered active navigation goal; "
                "replanning once: %s",
                active_goal_recovery_detail,
            )
            threading.Thread(
                target=self._run_navigation,
                args=(
                    decision_id,
                    dict(restart["candidate"]),
                    int(restart["start_goal_option_index"]),
                    list(restart["interaction_approach_attempts"]),
                ),
                daemon=True,
            ).start()
            return
        if active_goal_recovery_detail:
            detail = {**detail, "active_goal_recovery": active_goal_recovery_detail}
        # A failed rear-goal safety gate already means "replan/no motion".
        # Do not fall through to the legacy ungated stuck-recovery backoff.
        rear_safe_failure = bool(detail.get("rear_goal_recovery")) or str(
            detail.get("reason") or ""
        ).startswith("rear_goal_")
        recovery_detail = (
            {}
            if rear_safe_failure
            else self._maybe_run_stuck_recovery(decision_id, success, detail)
        )
        if recovery_detail:
            detail = {**detail, **recovery_detail}
        with self.lock:
            if self.selection is None or str(self.selection.get("decision_id") or "") != decision_id:
                return
            if (
                not success
                and self.machine.candidate is not None
                and str(self.machine.candidate.get("behavior_type") or "").upper()
                == "INTERACT"
            ):
                # This is executor-navigation failure, not a statement about
                # the object's physical state.  The decision node uses it for
                # bounded, episode-local approach reachability memory.
                detail.setdefault(
                    "failure_stage", "interaction_approach_navigation"
                )
                approach_reason = str(
                    detail.get("failure_reason") or detail.get("reason") or ""
                ).casefold()
                if approach_reason in {
                    "make_plan_unreachable",
                    "navigation_stagnation",
                    "navigation_timeout",
                    "navigation_terminal_failure",
                    "final_yaw_alignment_failed",
                    "interaction_pose_poll_exhausted",
                    "interaction_pose_invalid",
                    "visual_reposition_required",
                }:
                    # Exhausting a finite set of *poses* says nothing about
                    # the object's physical state.  The decision node records
                    # candidate-local approach memory for this normalized
                    # reason and deliberately avoids a target-wide cooldown.
                    detail.setdefault(
                        "interaction_approach_terminal_reason", approach_reason
                    )
                    detail["reason"] = "interaction_approach_options_exhausted"
                    detail["failure_reason"] = (
                        "interaction_approach_options_exhausted"
                    )
                    detail["failure_stage"] = "interaction_approach_exhausted"
                elif approach_reason == "interaction_approach_options_exhausted":
                    # A previous fallback branch has already consumed every
                    # approach. Preserve that stronger terminal contract rather
                    # than treating the feedback as one more ordinary approach
                    # failure in the decision layer.
                    detail["failure_stage"] = "interaction_approach_exhausted"
            if (
                success
                and self.machine.candidate is not None
                and str(self.machine.candidate.get("behavior_type") or "").upper()
                == "INTERACT"
            ):
                effective_pose = list(
                    detail.get("effective_interaction_approach_pose_xyyaw") or []
                )
                if len(effective_pose) >= 2:
                    bound_candidate = candidate_with_effective_interaction_approach(
                        self.machine.candidate,
                        effective_pose,
                        goal_option_index=int(
                            detail.get("interaction_approach_goal_option_index", 0)
                            or 0
                        ),
                        attempts=list(
                            detail.get("interaction_approach_attempts") or []
                        ),
                    )
                    self.machine.candidate = bound_candidate
                    # Keep executor-side feedback and bridge command metadata
                    # coherent with the exact fallback that move_base reached.
                    self.selection = dict(bound_candidate)
            wait_for_drawer_scan = bool(
                success and self._needs_fresh_drawer_scan_locked()
            )
            commands = self.machine.on_navigation_result(
                success,
                detail=detail,
                wait_for_drawer_scan=wait_for_drawer_scan,
            )
            if (
                self.machine.state == STATE_VERIFYING
                and requires_graph_verification(
                    self.ablation.module3, self.selection
                )
            ):
                commands.extend(self._verify_graph_locked())
        self._dispatch(commands)

    def _maybe_run_stuck_recovery(
        self, decision_id: str, success: bool, detail: dict
    ) -> dict:
        if success:
            self._reset_stuck_failures()
            return {}
        if not self.stuck_recovery_enabled:
            return {}
        if not is_stuck_recovery_failure(detail):
            self._reset_stuck_failures()
            return {}
        pose = self._current_pose(self.map_frame)
        if pose is None:
            return {}
        candidate_id = ""
        with self.lock:
            if self.selection is not None:
                candidate_id = str(self.selection.get("candidate_id") or "")
        if self._stuck_failure_origin_xy is None:
            self._stuck_failure_origin_xy = (pose[0], pose[1])
            self._stuck_failure_candidate_ids = {candidate_id} if candidate_id else set()
            return {}
        displacement = math.hypot(
            pose[0] - self._stuck_failure_origin_xy[0],
            pose[1] - self._stuck_failure_origin_xy[1],
        )
        if displacement >= self.stuck_recovery_min_displacement_m:
            self._stuck_failure_origin_xy = (pose[0], pose[1])
            self._stuck_failure_candidate_ids = {candidate_id} if candidate_id else set()
            return {}
        if candidate_id:
            self._stuck_failure_candidate_ids.add(candidate_id)
        if len(self._stuck_failure_candidate_ids) < self.stuck_recovery_subgoal_failures:
            return {}
        backed_off = self._drive_linear_recovery(
            decision_id, -abs(self.stuck_recovery_speed_mps), self.stuck_recovery_backoff_distance_m
        )
        escaped = False
        if not backed_off:
            escaped = self._escape_nearest_obstacle(decision_id)
        self._reset_stuck_failures()
        return {
            "stuck_recovery": "backoff" if backed_off else "obstacle_escape" if escaped else "failed",
            "stuck_failure_count": self.stuck_recovery_subgoal_failures,
        }

    def _reset_stuck_failures(self) -> None:
        self._stuck_failure_origin_xy = None
        self._stuck_failure_candidate_ids.clear()

    def _drive_linear_recovery(
        self, decision_id: str, linear_x: float, distance_m: float
    ) -> bool:
        start = self._current_pose(self.map_frame)
        if start is None:
            return False
        target_distance = self._safe_recovery_distance(start, linear_x, distance_m)
        if target_distance < self.stuck_recovery_min_displacement_m:
            return False
        deadline = time.monotonic() + self.stuck_recovery_timeout_s
        try:
            while (
                not rospy.is_shutdown()
                and self._navigation_is_current(decision_id)
                and time.monotonic() < deadline
            ):
                pose = self._current_pose(self.map_frame)
                traveled = 0.0 if pose is None else math.hypot(
                    pose[0] - start[0], pose[1] - start[1]
                )
                if traveled >= max(0.0, target_distance - 0.02):
                    return True
                remaining = target_distance - traveled
                if pose is None or (
                    remaining > 0.05
                    and self._safe_recovery_distance(pose, linear_x, remaining) < 0.05
                ):
                    return False
                command = Twist()
                command.linear.x = float(linear_x)
                self.cmd_vel_pub.publish(command)
                time.sleep(0.05)
        finally:
            self.cmd_vel_pub.publish(Twist())
        pose = self._current_pose(self.map_frame)
        return bool(
            pose is not None
            and math.hypot(pose[0] - start[0], pose[1] - start[1])
            >= max(0.0, target_distance - 0.02)
        )

    def _safe_recovery_distance(
        self, pose: tuple[float, float, float], linear_x: float, requested_distance_m: float
    ) -> float:
        with self.lock:
            occupancy = self._latest_occupancy
        if occupancy is None or not occupancy.data:
            return 0.0
        info = occupancy.info
        return safe_grid_motion_distance(
            occupancy.data,
            int(info.width),
            int(info.height),
            float(info.resolution),
            (float(info.origin.position.x), float(info.origin.position.y)),
            pose,
            1.0 if float(linear_x) >= 0.0 else -1.0,
            requested_distance_m,
            self.stuck_recovery_robot_radius_m,
            self.stuck_recovery_safety_margin_m,
            unknown_is_blocked=self.stuck_recovery_unknown_is_blocked,
        )

    def _escape_nearest_obstacle(self, decision_id: str) -> bool:
        pose = self._current_pose(self.map_frame)
        with self.lock:
            occupancy = self._latest_occupancy
        if pose is None or occupancy is None or not occupancy.data:
            return False
        info = occupancy.info
        resolution = float(info.resolution)
        if resolution <= 0.0:
            return False
        nearest = None
        for index, value in enumerate(occupancy.data):
            if int(value) < 50:
                continue
            column, row = index % int(info.width), index // int(info.width)
            x = float(info.origin.position.x) + (column + 0.5) * resolution
            y = float(info.origin.position.y) + (row + 0.5) * resolution
            distance = math.hypot(x - pose[0], y - pose[1])
            if nearest is None or distance < nearest[0]:
                nearest = (distance, x, y)
        if nearest is None:
            return False
        away_yaw = math.atan2(pose[1] - nearest[2], pose[0] - nearest[1])
        if not self._rotate_to_yaw(
            decision_id, self.map_frame, away_yaw, 0.20,
            self.rear_goal_rotate_speed_rad_s, self.stuck_recovery_timeout_s,
        ):
            return False
        return self._drive_linear_recovery(
            decision_id, abs(self.stuck_recovery_speed_mps),
            self.stuck_recovery_obstacle_escape_distance_m,
        )

    def _navigation_is_current(self, decision_id: str) -> bool:
        with self.lock:
            return bool(
                self.selection is not None
                and str(self.selection.get("decision_id") or "") == decision_id
                and self.machine.state in {STATE_NAVIGATING, STATE_APPROACH_INTERACTION}
            )

    def _finish_terminal(self, command: dict) -> None:
        with self.lock:
            selection = dict(self.selection or {})
            decision_id = str(selection.get("decision_id") or "")
            was_navigating = self.machine.state in {
                STATE_NAVIGATING,
                STATE_APPROACH_INTERACTION,
            }
            status = "SUCCEEDED" if command.get("success") else "FAILED"
            detail = dict(command.get("detail") or {})
            if decision_id:
                self._navigation_failure_recovery_attempts.pop(decision_id, None)
            drawer_scan_wait = self._drawer_scan_wait_records.pop(decision_id, None)
            self._drawer_scan_wait_contexts.pop(decision_id, None)
            if drawer_scan_wait:
                detail.setdefault("drawer_scan_wait", drawer_scan_wait)
            if self.model_events:
                detail["mllm_events"] = list(self.model_events)
            self._publish_feedback(selection, status, bool(command.get("success")), detail)
            self.selection = None
            self._clear_drawer_scan_execution_wait_locked()
            self.machine.reset()
        if was_navigating:
            self.move_base.cancel_goal()

    def _publish_feedback(
        self, selection: dict, status: str, success: bool | None, detail: dict
    ) -> None:
        payload = {
            "decision_id": selection.get("decision_id", ""),
            "candidate_id": selection.get("candidate_id", ""),
            "behavior_type": selection.get("behavior_type", ""),
            "target_id": selection.get("target_id", ""),
            "target_name": selection.get("target_name", ""),
            "command_id": self._command_id(selection),
            "status": status,
            "success": success,
            "detail": detail,
            "timestamp": time.time(),
        }
        self.feedback_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )

    def _matches_active(self, payload: dict) -> bool:
        if self.selection is None:
            return False
        command_id = str(payload.get("command_id") or "")
        candidate_id = str(payload.get("candidate_id") or "")
        return command_id == self._command_id(self.selection) or (
            candidate_id and candidate_id == str(self.selection.get("candidate_id") or "")
        )

    @staticmethod
    def _command_id(candidate: dict) -> str:
        return f"{candidate.get('decision_id', 'decision')}:{candidate.get('candidate_id', 'candidate')}"


if __name__ == "__main__":
    SemanticBehaviorExecutor()
    rospy.spin()
