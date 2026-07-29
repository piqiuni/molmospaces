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
    STATE_WAITING_FOR_DRAWER_SCAN,
    STATE_VERIFYING,
    bounded_empty_plan_retry_delay,
    committed_turn_sign,
    is_post_interaction_traversal_navigation,
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
    prerotation_rgb_step_gate,
    requires_graph_verification,
    is_stuck_recovery_failure,
    safe_grid_motion_distance,
)
from semantic_decision_py_pkg.ros_compat import patch_roslogging_findcaller_for_py311
from semantic_decision_py_pkg.visual_interaction_planning import (
    action_for_opaque_open_contract,
    candidate_with_direct_drawer_scan,
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
    validate_visual_interaction_plan,
    validate_visual_verification,
)

patch_roslogging_findcaller_for_py311()

import actionlib
import rospy
import tf
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
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
        model_name = str(model_config.get("model", "") or "")
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
            64,
            int(model_config.get("verification_max_output_tokens", min(self.mllm_client.config.max_tokens, 192))),
        )
        self.skill_timeout_s = max(
            0.1, float(model_config.get("skill_timeout_s", 4.0))
        )
        self.verification_timeout_s = max(
            0.1, float(model_config.get("verification_timeout_s", 4.0))
        )
        self.mllm_crop_margin_ratio = max(
            0.0, float(model_config.get("crop_margin_ratio", 0.10))
        )
        self.mllm_crop_max_side_px = max(
            128, int(model_config.get("crop_max_side_px", 512))
        )
        self.drawer_scan_wait_timeout_s = max(
            0.1, float(config.get("drawer_scan_wait_timeout_s", 8.0))
        )
        self.drawer_scan_wait_poll_interval_s = max(
            0.01, float(config.get("drawer_scan_wait_poll_interval_s", 0.10))
        )
        self.machine = BehaviorExecutionStateMachine(
            ExecutionConfig(
                navigation_timeout_s=float(config.get("navigation_timeout_s", 180.0)),
                interaction_navigation_timeout_s=float(
                    config.get("interaction_navigation_timeout_s", 180.0)
                ),
                interaction_timeout_s=float(config.get("interaction_timeout_s", 30.0)),
                drawer_scan_wait_timeout_s=self.drawer_scan_wait_timeout_s,
                verification_timeout_s=float(config.get("verification_timeout_s", 30.0)),
                explore_prepare_timeout_s=float(
                    config.get("explore_prepare_timeout_s", 10.0)
                ),
                explore_finalize_timeout_s=float(
                    config.get("explore_finalize_timeout_s", 10.0)
                ),
            )
        )
        self.map_frame = str(config.get("map_frame", "tf_frame_map"))
        self.base_frame = str(config.get("base_frame", "tf_frame_base_link"))
        self.rear_goal_prerotate_enabled = bool(
            config.get("rear_goal_prerotate_enabled", True)
        )
        self.rear_goal_enter_angle_rad = float(
            config.get("rear_goal_enter_angle_rad", 1.75)
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
            config.get("rear_goal_prerotate_step_sync_enabled", False)
        )
        self.rear_goal_prerotate_control_dt_s = max(
            1e-3,
            float(config.get("rear_goal_prerotate_control_dt_s", 0.2)),
        )
        self.rear_goal_prerotate_max_control_steps = max(
            1,
            int(config.get("rear_goal_prerotate_max_control_steps", 12)),
        )
        self.rear_goal_prerotate_step_sync_stall_timeout_s = max(
            0.1,
            float(config.get("rear_goal_prerotate_step_sync_stall_timeout_s", 2.0)),
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
        self.final_align_enabled = bool(config.get("final_align_enabled", True))
        self.final_align_max_distance_m = float(
            config.get("final_align_max_distance_m", config.get("interaction_final_align_max_distance_m", 0.12))
        )
        self.final_align_yaw_tolerance_rad = float(
            config.get("final_align_yaw_tolerance_rad", config.get("interaction_final_align_yaw_tolerance_rad", 0.15))
        )
        self.final_align_rotate_speed_rad_s = float(
            config.get("final_align_rotate_speed_rad_s", config.get("interaction_final_align_rotate_speed_rad_s", 0.30))
        )
        self.final_align_trigger_delay_s = float(
            config.get("final_align_trigger_delay_s", config.get("interaction_final_align_trigger_delay_s", 2.0))
        )
        self.final_align_timeout_s = float(
            config.get("final_align_timeout_s", config.get("interaction_final_align_timeout_s", 15.0))
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
            config.get("stuck_recovery_robot_radius_m", 0.30)
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
            int(config.get("interaction_approach_fallback_max_attempts", 3)),
        )
        self.interaction_approach_fallback_cancel_wait_s = max(
            0.0,
            float(config.get("interaction_approach_fallback_cancel_wait_s", 0.5)),
        )
        self.lock = threading.RLock()
        self.selection: dict | None = None
        self.latest_graph: dict = {}
        self._last_explore_reservation_publish_at = 0.0
        self._explore_reservation_publish_count = 0
        self._explore_feedback_received_count = 0
        self._explore_feedback_matched_count = 0
        self._explore_feedback_ignored_count = 0
        self._last_explore_feedback = {}
        self._stuck_failure_origin_xy: tuple[float, float] | None = None
        self._stuck_failure_candidate_ids: set[str] = set()
        self._latest_occupancy: OccupancyGrid | None = None
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
        self._latest_rgb_step_seq: int | None = None
        self._latest_rgb_step_received_at = 0.0
        self.latest_image = None
        self.latest_image_sequence = 0
        self.pre_interaction_image_sequence = 0
        self._drawer_scan_wait_contexts: dict[str, dict] = {}
        self._drawer_scan_wait_records: dict[str, dict] = {}
        self.active_skill_plan: dict = {}
        self.pending_skill_actions: list[dict] = []
        self.interaction_command_sequence = 0
        self.verification_retries = 0
        self.model_events: list[dict] = []
        self.feedback_pub = rospy.Publisher(
            topics.get("behavior_feedback", "/semantic_decision/behavior_feedback"),
            String,
            queue_size=10,
            latch=True,
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
            topics.get("interaction_result", "/semantic_mapping/interaction_result"),
            String,
            self._interaction_result_callback,
            queue_size=10,
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
        if self.rear_goal_prerotate_step_sync_enabled:
            rospy.Subscriber(
                topics.get("step_sync", "/molmo_spaces/step_sync"),
                String,
                self._step_sync_callback,
                queue_size=1,
            )
        self.timer = rospy.Timer(rospy.Duration(0.2), self._tick)

    def _selection_callback(self, message: String) -> None:
        try:
            selection = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            if self.machine.state != STATE_IDLE or self.selection is not None:
                self._publish_feedback(selection, "REJECTED", False, {"reason": "executor_busy"})
                return
            self.selection = selection
            self.active_skill_plan = {}
            self.pending_skill_actions = []
            self.interaction_command_sequence = 0
            self.verification_retries = 0
            self.model_events = []
            decision_id = str(selection.get("decision_id") or "")
            if decision_id:
                self._drawer_scan_wait_contexts.pop(decision_id, None)
                self._drawer_scan_wait_records.pop(decision_id, None)
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

    def _interaction_result_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            if not self._matches_active(payload):
                return
            self._record_post_interaction_costmap_baseline_locked(payload)
            visual_plan = dict(self.active_skill_plan.get("visual_operation_plan") or {})
            is_drawer_scan = str(visual_plan.get("target_type") or "") == "drawer_container"
            if (
                self.ablation.module3 == "mllm_skill_verified"
                and bool(payload.get("success"))
                and self.pending_skill_actions
            ):
                next_action = self.pending_skill_actions.pop(0)
                next_candidate = self._candidate_for_skill_action(
                    dict(self.selection or {}), next_action
                )
                commands = []
            else:
                next_candidate = None
                commands = self.machine.on_interaction_result(
                    bool(payload.get("success")), detail=payload
                )
            if (
                self.machine.state == STATE_VERIFYING
                and self.ablation.module3 == "direct_atomic"
            ):
                commands.extend(
                    self.machine.on_verification_result(
                        bool(payload.get("success")),
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
                if self.evaluator_opaque_open_only and bool(payload.get("success")):
                    # The V3 evaluator exposes interaction as a sealed semantic
                    # skill: success is the public postcondition contract.  A
                    # second visual classifier can produce false negatives and
                    # retry an already-open object, while no joint state is
                    # available (or needed) outside the sealed skill.
                    commands.extend(
                        self.machine.on_verification_result(
                            True,
                            detail={
                                **payload,
                                "verification_mode": "evaluator_skill_postcondition",
                            },
                        )
                    )
                elif is_drawer_scan:
                    # A drawer scan intentionally closes each drawer after its
                    # low-view observation.  Re-checking the final exterior
                    # crop as though it should remain open would reject a
                    # successful scan.  The semantic map receives the frames
                    # captured while each drawer is open instead.
                    commands.extend(
                        self.machine.on_verification_result(
                            True,
                            detail={
                                **payload,
                                "verification_mode": "drawer_scan_backend",
                            },
                        )
                    )
                else:
                    decision_id = str((self.selection or {}).get("decision_id") or "")
                    result_image_sequence = self.latest_image_sequence
                    threading.Thread(
                        target=self._run_visual_verification,
                        args=(decision_id, payload, result_image_sequence),
                        daemon=True,
                    ).start()
        if next_candidate is not None:
            self._publish_interaction_command(next_candidate)
        else:
            self._dispatch(commands)

    def _step_sync_callback(self, message: String) -> None:
        """Record evaluator progress without using TF's keepalive stream."""

        try:
            payload = json.loads(message.data)
            step_index = int(payload["step_index"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        with self.lock:
            self._latest_step_sync_index = step_index
            self._latest_step_sync_received_at = time.monotonic()

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
            reason = self.machine.timeout_reason()
            if reason in {"navigation_timeout", "interaction_navigation_timeout"}:
                reason = ""
            cancel_navigation = bool(reason) and self.machine.state in {
                STATE_NAVIGATING,
                STATE_APPROACH_INTERACTION,
            }
            if reason == "drawer_scan_fresh_frame_timeout":
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
                "drawer_scan_wait": dict(
                    self._drawer_scan_wait_contexts.get(
                        str((self.selection or {}).get("decision_id") or ""),
                        {},
                    )
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
            elif kind == "navigate":
                candidate = dict(command["candidate"])
                decision_id = str(candidate.get("decision_id") or "")
                threading.Thread(
                    target=self._run_navigation,
                    args=(decision_id, candidate),
                    daemon=True,
                ).start()
            elif kind == "interact":
                if self.ablation.module3 == "mllm_skill_verified":
                    candidate = dict(command["candidate"])
                    decision_id = str(candidate.get("decision_id") or "")
                    threading.Thread(
                        target=self._plan_and_publish_interaction,
                        args=(decision_id, candidate),
                        daemon=True,
                    ).start()
                else:
                    self._publish_interaction_command(command["candidate"])
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

    def _publish_interaction_command(self, candidate: dict) -> None:
        interaction = candidate.get("interaction_command") or {}
        metadata = candidate.get("metadata") or {}
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
            "approach_goal_xyyaw": list(candidate.get("goal_xyyaw") or []),
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
        if str(interaction.get("sequence_type") or "").casefold() == "drawer_scan":
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
            image_data = self._object_crop_data_locked(node)
        context = visual_interaction_planning_context(
            object_id=str(candidate.get("target_id") or ""),
            object_name=str(candidate.get("target_name") or ""),
            expected_target_type=expected_target_type,
            requested_action=requested_action,
        )
        response = self.mllm_client.request_json(
            role="skill_planning",
            instruction=VISUAL_INTERACTION_PLANNING_INSTRUCTION,
            context=context,
            images=[image_data] if image_data else [],
            timeout_s=self.skill_timeout_s,
            max_tokens=self.skill_max_output_tokens,
            metrics_context=self._model_metrics_context(
                "skill_planning", decision_id, candidate
            ),
        )
        planned_candidate = dict(candidate)
        result_source = "rule_fallback_model_error"
        if response.payload is not None and not response.error:
            try:
                plan = validate_visual_interaction_plan(
                    response.payload,
                    expected_target_type=expected_target_type,
                    requested_action=requested_action,
                )
                planned_candidate = candidate_with_visual_operation_plan(
                    planned_candidate, plan
                )
                result_source = "model"
                with self.lock:
                    self.active_skill_plan = {
                        "visual_operation_plan": plan,
                        "subactions": [],
                        "max_retries": 0,
                    }
                    self.pending_skill_actions = []
            except ValueError:
                result_source = "rule_fallback_invalid_response"
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
        response = self.mllm_client.request_json(
            role="visual_verification",
            instruction=(
                "Inspect the current cropped image of the interaction target after execution. "
                "Determine whether it now matches the requested state using the expected action "
                "and visible evidence only. Return exactly one compact JSON object with success, "
                "confidence, reason, observed_states, new_contents_visible, and retry_action. "
                "Use a reason no longer than twelve words; do not output markdown or extra fields."
            ),
            context={
                "target": {
                    "id": str(selection.get("target_id") or ""),
                    "name": str(selection.get("target_name") or ""),
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
                "pre_interaction_skill": {
                    "subactions": list(self.active_skill_plan.get("subactions") or []),
                },
            },
            images=[after] if after else [],
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

        if self.ablation.module3 != "mllm_skill_verified" or self.selection is None:
            return False
        if str(self.selection.get("behavior_type") or "").upper() != "INTERACT":
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

    def _object_crop_data_locked(self, node: dict) -> str:
        if self.latest_image is None:
            return ""
        attributes = node.get("attributes") or {}
        box = (
            attributes.get("projected_bbox_2d")
            or node.get("projected_bbox_2d")
            or attributes.get("bbox_2d")
            or node.get("bbox_2d")
        )
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            return ""
        image = self.latest_image
        height, width = image.shape[:2]
        raw_x0, raw_y0, raw_x1, raw_y1 = [
            int(round(float(value))) for value in box[:4]
        ]
        left, right = sorted((raw_x0, raw_x1))
        top, bottom = sorted((raw_y0, raw_y1))
        margin_x = int(round((right - left) * self.mllm_crop_margin_ratio))
        margin_y = int(round((bottom - top) * self.mllm_crop_margin_ratio))
        left = max(0, min(width - 1, left - margin_x))
        right = min(width, max(left + 1, right + margin_x))
        top = max(0, min(height - 1, top - margin_y))
        bottom = min(height, max(top + 1, bottom + margin_y))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            return ""
        crop_height, crop_width = crop.shape[:2]
        scale = min(
            1.0,
            float(self.mllm_crop_max_side_px) / max(crop_width, crop_height),
        )
        if scale < 1.0:
            crop = cv2.resize(
                crop,
                (
                    max(1, int(round(crop_width * scale))),
                    max(1, int(round(crop_height * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(
            ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")

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
    ) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        committed_sign = turn_sign
        with self.lock:
            last_step_sync_index = self._latest_step_sync_index
            last_step_sync_at = self._latest_step_sync_received_at
            start_rgb_step_seq = self._latest_rgb_step_seq
            last_sent_rgb_step_seq = start_rgb_step_seq
            start_rgb_received_at = self._latest_rgb_step_received_at
        nonzero_commands_sent = 0
        saw_new_rgb_step = False
        last_rgb_step_advance_at = time.monotonic()
        last_step_sync_progress_at = time.monotonic()
        gate_stall_timeout_s = max(
            0.1, float(step_sync_stall_timeout_s or 2.0)
        )

        def finish_prerotation(reason: str, success: bool) -> bool:
            if max_prerotate_control_steps is not None:
                rospy.loginfo(
                    "[semantic_behavior_executor] pre-rotation finished "
                    "reason=%s nonzero_commands=%d start_rgb_seq=%s last_rgb_seq=%s",
                    reason,
                    nonzero_commands_sent,
                    start_rgb_step_seq,
                    last_sent_rgb_step_seq,
                )
            return success

        try:
            while (
                not rospy.is_shutdown()
                and self._navigation_is_current(decision_id)
                and time.monotonic() < deadline
            ):
                current_rgb_step_seq = None
                if max_prerotate_control_steps is not None:
                    with self.lock:
                        current_step_sync_index = self._latest_step_sync_index
                        current_step_sync_at = self._latest_step_sync_received_at
                        current_rgb_step_seq = self._latest_rgb_step_seq
                    if current_step_sync_at > last_step_sync_at:
                        if (
                            last_step_sync_index is not None
                            and (
                                current_step_sync_index is None
                                or current_step_sync_index < last_step_sync_index
                            )
                        ):
                            return finish_prerotation("sync_reset", False)
                        last_step_sync_index = current_step_sync_index
                        last_step_sync_at = current_step_sync_at
                        last_step_sync_progress_at = time.monotonic()
                    now = time.monotonic()
                    if not saw_new_rgb_step and (
                        start_rgb_received_at <= 0.0
                        or now - start_rgb_received_at
                        >= gate_stall_timeout_s
                    ):
                        return finish_prerotation("rgb_stale_at_start", False)
                    if current_rgb_step_seq is not None:
                        if (
                            last_sent_rgb_step_seq is not None
                            and current_rgb_step_seq < last_sent_rgb_step_seq
                        ):
                            return finish_prerotation("rgb_reset", False)
                        if (
                            last_sent_rgb_step_seq is None
                            or current_rgb_step_seq > last_sent_rgb_step_seq
                        ):
                            saw_new_rgb_step = True
                            last_rgb_step_advance_at = now
                    if (
                        nonzero_commands_sent > 0
                        and now - last_step_sync_progress_at
                        >= gate_stall_timeout_s
                    ):
                        return finish_prerotation("step_sync_stall", False)
                    if (
                        now - last_rgb_step_advance_at
                        >= gate_stall_timeout_s
                    ):
                        return finish_prerotation("rgb_step_stall", False)
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
                if max_prerotate_control_steps is not None:
                    gate = prerotation_rgb_step_gate(
                        last_sent_rgb_step_seq=last_sent_rgb_step_seq,
                        current_rgb_step_seq=current_rgb_step_seq,
                        nonzero_commands_sent=nonzero_commands_sent,
                        max_control_steps=max_prerotate_control_steps,
                    )
                    if gate == "stop":
                        return finish_prerotation("rgb_step_budget", False)
                    if gate == "wait":
                        time.sleep(0.01)
                        continue
                    last_sent_rgb_step_seq = current_rgb_step_seq
                    nonzero_commands_sent += 1
                self._publish_rotation(float(committed_sign) * abs(float(speed_rad_s)))
                time.sleep(0.05)
        finally:
            self._publish_rotation(0.0)
        return False

    def _prerotate_for_rear_goal(
        self,
        decision_id: str,
        frame_id: str,
        goal_x: float,
        goal_y: float,
        heading_target_xy: tuple[float, float] | None = None,
    ) -> bool:
        if not self.rear_goal_prerotate_enabled:
            return True
        # A navigation pre-turn must follow the first reachable segment of the
        # global path.  When preflight did not produce a valid path (for
        # example an EXPLORE fail-open), using the final goal can turn the base
        # away from the path that move_base will eventually choose.
        heading_target_xy = navigation_prerotation_heading_target(heading_target_xy)
        if heading_target_xy is None:
            rospy.loginfo(
                "[semantic_behavior_executor] skipped rear-goal prerotation: "
                "no valid path lookahead"
            )
            return True
        pose = self._current_pose(frame_id)
        if pose is None:
            return True
        heading_x, heading_y = heading_target_xy
        target_yaw = math.atan2(heading_y - pose[1], heading_x - pose[0])
        error = normalize_angle(target_yaw - pose[2])
        if abs(error) <= self.rear_goal_enter_angle_rad:
            return True
        turn_sign = committed_turn_sign(
            error,
            self.rear_goal_pi_tie_tolerance_rad,
            self.rear_goal_pi_turn_sign,
        )
        max_prerotate_control_steps = None
        step_sync_stall_timeout_s = None
        if self.rear_goal_prerotate_step_sync_enabled:
            max_prerotate_control_steps = prerotation_control_step_budget(
                error,
                self.rear_goal_exit_angle_rad,
                self.rear_goal_rotate_speed_rad_s,
                self.rear_goal_prerotate_control_dt_s,
                self.rear_goal_prerotate_max_control_steps,
            )
            step_sync_stall_timeout_s = self.rear_goal_prerotate_step_sync_stall_timeout_s
            if max_prerotate_control_steps <= 0:
                return True
        return self._rotate_to_yaw(
            decision_id,
            frame_id,
            target_yaw,
            self.rear_goal_exit_angle_rad,
            self.rear_goal_rotate_speed_rad_s,
            self.rear_goal_prerotate_timeout_s,
            turn_sign=turn_sign,
            max_prerotate_control_steps=max_prerotate_control_steps,
            step_sync_stall_timeout_s=step_sync_stall_timeout_s,
        )

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
                    "interaction_approach_attempts": interaction_approach_attempts,
                },
            )
            return
        metadata = candidate.get("metadata") or {}
        interaction = candidate.get("interaction_command") or {}
        if behavior_type == "INTERACT":
            direct_distance_tolerance = float(
                interaction.get("interaction_ready_distance_m", 0.45) or 0.45
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
                    self._handle_navigation_result(
                        decision_id,
                        True,
                        {
                            "reason": "already_at_verified_approach_pose",
                            "position_error_m": position_error,
                            "yaw_error_rad": yaw_error,
                        },
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
            selected_attempt = dict(attempted_goals[-1])
            selected_attempt["navigation_attempt"] = (
                len(interaction_approach_attempt_history) + 1
            )
            interaction_approach_attempt_history.append(selected_attempt)
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
                "[semantic_behavior_executor] rear-goal prerotation timed out; sending move_base goal"
            )
        if not self._navigation_is_current(decision_id):
            return
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.z = math.sin(0.5 * yaw)
        goal.target_pose.pose.orientation.w = math.cos(0.5 * yaw)
        self.move_base.send_goal(goal)
        navigation_started_at = time.monotonic()
        start_pose = self._current_pose(goal_frame)
        start_goal_distance_m = (
            None
            if start_pose is None
            else math.hypot(x - start_pose[0], y - start_pose[1])
        )
        progress_watchdog = NavigationProgressWatchdog(
            timeout_s=self.navigation_stagnation_timeout_s,
            min_displacement_m=self.navigation_stagnation_distance_m,
            min_yaw_change_rad=self.navigation_stagnation_yaw_rad,
            min_goal_distance_reduction_m=(
                self.navigation_stagnation_goal_distance_reduction_m
            ),
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
                        self._handle_navigation_result(
                            decision_id,
                            bool(aligned),
                            {
                                "reason": "direct_final_yaw_alignment",
                                "position_error_m": distance,
                                "yaw_error_rad": yaw_error,
                            },
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
                    self._handle_navigation_result(
                        decision_id,
                        bool(aligned),
                        {"reason": "navigation_timeout_final_alignment"},
                    )
                    return
            self._handle_navigation_result(
                decision_id,
                False,
                {
                    "reason": "navigation_timeout",
                    "interaction_approach_attempts": interaction_approach_attempt_history,
                },
            )
            return
        success = state == GoalStatus.SUCCEEDED
        detail = {
            "status_code": state,
            "status": self.move_base.get_goal_status_text() or str(state),
        }
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
        self._handle_navigation_result(decision_id, success, detail)

    def _retry_interaction_approach(
        self,
        decision_id: str,
        candidate: dict,
        selected_option_index: int | None,
        interaction_approach_attempts: list[dict],
        goal_option_count: int,
        failure_detail: dict,
    ) -> bool:
        if selected_option_index is None:
            return False
        next_option_index = next_interaction_approach_option_index(
            behavior_type=str(candidate.get("behavior_type") or ""),
            failure_detail=failure_detail,
            selected_option_index=selected_option_index,
            attempted_navigation_count=len(interaction_approach_attempts),
            max_navigation_attempts=self.interaction_approach_fallback_max_attempts,
            goal_option_count=goal_option_count,
        )
        if next_option_index is None:
            return False
        attempts = [dict(attempt) for attempt in interaction_approach_attempts]
        if attempts:
            attempts[-1]["outcome"] = str(failure_detail.get("reason") or "failed")
            attempts[-1]["failure_detail"] = {
                "goal_distance_m": failure_detail.get("goal_distance_m"),
                "local_plan_fresh": failure_detail.get("local_plan_fresh"),
            }
        if not self._navigation_is_current(decision_id):
            return True
        if self.interaction_approach_fallback_cancel_wait_s > 0.0:
            self.move_base.wait_for_result(
                rospy.Duration(self.interaction_approach_fallback_cancel_wait_s)
            )
        if not self._navigation_is_current(decision_id):
            return True
        rospy.logwarn(
            "[semantic_behavior_executor] INTERACT approach stagnated; "
            "retrying option %d/%d (attempt %d/%d)",
            next_option_index + 1,
            goal_option_count,
            len(attempts) + 1,
            self.interaction_approach_fallback_max_attempts,
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

    def _handle_navigation_result(
        self, decision_id: str, success: bool, detail: dict
    ) -> None:
        recovery_detail = self._maybe_run_stuck_recovery(decision_id, success, detail)
        if recovery_detail:
            detail = {**detail, **recovery_detail}
        with self.lock:
            if self.selection is None or str(self.selection.get("decision_id") or "") != decision_id:
                return
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
            drawer_scan_wait = self._drawer_scan_wait_records.pop(decision_id, None)
            self._drawer_scan_wait_contexts.pop(decision_id, None)
            if drawer_scan_wait:
                detail.setdefault("drawer_scan_wait", drawer_scan_wait)
            if self.model_events:
                detail["mllm_events"] = list(self.model_events)
            self._publish_feedback(selection, status, bool(command.get("success")), detail)
            self.selection = None
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
