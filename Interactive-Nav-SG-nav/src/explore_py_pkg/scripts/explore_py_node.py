#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path


def _patch_roslogging_findcaller_for_py311() -> None:
    if sys.version_info < (3, 11):
        return
    try:
        import logging
        import rosgraph.roslogging
    except Exception:
        return

    def find_caller(self, stack_info=False, stacklevel=1):  # noqa: ANN001
        frame = logging.currentframe()
        if frame is not None:
            frame = frame.f_back
        while frame and stacklevel > 1:
            frame = frame.f_back
            stacklevel -= 1
        if frame is None:
            return "(unknown file)", 0, "(unknown function)", None
        code = frame.f_code
        return code.co_filename, frame.f_lineno, code.co_name, None

    rosgraph.roslogging.RospyLogger.findCaller = find_caller


_patch_roslogging_findcaller_for_py311()

import rospy
from actionlib_msgs.msg import GoalID, GoalStatusArray
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from nav_msgs.srv import GetPlan
from std_msgs.msg import Empty, String
from visualization_msgs.msg import Marker, MarkerArray

from explore_py_pkg.frontier_core import FrontierConfig, FrontierExplorerCore, GridSpec, OccupancyGridData
from explore_py_pkg.nav_client import TERMINAL_FAILURE, TERMINAL_SUCCESS, status_name
from explore_py_pkg.skill_api import ExplorationSkillApi
from explore_py_pkg.state import ExplorerState, ExplorerStateConfig, SUBGOAL_REACHED, SUBGOAL_WAITING
from explore_py_pkg.value_maps import ValueMapFusion


class ExplorePyNode:
    def __init__(self):
        rospy.init_node("explore_py")
        self.topics = rospy.get_param("~topics", {}) or {}
        self.frames = rospy.get_param("~frames", {}) or {}
        exploration_cfg = rospy.get_param("~exploration", {}) or {}
        frontier_cfg = rospy.get_param("~frontier", {}) or {}
        scoring_cfg = rospy.get_param("~scoring", {}) or {}

        self.map_frame = self.frames.get("map_frame", "tf_frame_map")
        self.external_behavior_control = bool(
            exploration_cfg.get("external_behavior_control", False)
        )
        self.tick_rate_hz = float(exploration_cfg.get("tick_rate_hz", 1.0))
        self.goal_republish_interval_sec = float(exploration_cfg.get("goal_republish_interval_sec", 2.0))
        self.active_goal_frontier_min_cells = int(exploration_cfg.get("active_goal_frontier_min_cells", 1))
        self.frontier_gone_confirm_ticks = int(exploration_cfg.get("frontier_gone_confirm_ticks", 3))
        self.frontier_gone_min_goal_age_sec = float(exploration_cfg.get("frontier_gone_min_goal_age_sec", 8.0))
        self.initial_spin_enabled = bool(exploration_cfg.get("initial_spin_enabled", False))
        self.initial_spin_angle_rad = float(exploration_cfg.get("initial_spin_angle_rad", 2.0 * math.pi))
        self.initial_spin_angular_speed = float(exploration_cfg.get("initial_spin_angular_speed", 0.35))
        self.initial_spin_timeout_sec = float(exploration_cfg.get("initial_spin_timeout_sec", 25.0))
        self.initial_spin_settle_sec = float(exploration_cfg.get("initial_spin_settle_sec", 1.0))
        self.initial_spin_cmd_rate_hz = float(exploration_cfg.get("initial_spin_cmd_rate_hz", 10.0))
        self.initial_spin_done = not self.initial_spin_enabled
        self.initial_spin_active = False
        self.initial_spin_start_time = 0.0
        self.initial_spin_done_time = 0.0
        self.initial_spin_last_yaw = None
        self.initial_spin_accumulated_yaw = 0.0
        self.initial_spin_reason = "disabled" if self.initial_spin_done else "pending"
        self.local_plan_watchdog_enabled = bool(exploration_cfg.get("local_plan_watchdog_enabled", True))
        self.local_plan_watchdog_sec = float(exploration_cfg.get("local_plan_watchdog_sec", 12.0))
        self.local_plan_min_poses = int(exploration_cfg.get("local_plan_min_poses", 3))
        self.local_plan_min_length_m = float(exploration_cfg.get("local_plan_min_length_m", 0.20))
        self.global_plan_min_poses = int(exploration_cfg.get("global_plan_min_poses", 3))
        self.plan_freshness_sec = float(exploration_cfg.get("plan_freshness_sec", 4.0))
        self.global_plan_current_goal_check_enabled = bool(
            exploration_cfg.get("global_plan_current_goal_check_enabled", True)
        )
        self.global_plan_current_goal_grace_sec = float(
            exploration_cfg.get("global_plan_current_goal_grace_sec", 4.0)
        )
        self.external_navigation_plan_grace_sec = float(
            exploration_cfg.get("external_navigation_plan_grace_sec", 16.0)
        )
        self.global_plan_goal_tolerance_m = float(exploration_cfg.get("global_plan_goal_tolerance_m", 0.6))
        self.make_plan_preflight_enabled = bool(
            exploration_cfg.get("make_plan_preflight_enabled", True)
        )
        self.make_plan_service = str(
            exploration_cfg.get("make_plan_service", "/move_base/make_plan")
        )
        self.make_plan_service_wait_sec = float(
            exploration_cfg.get("make_plan_service_wait_sec", 2.0)
        )
        self.make_plan_tolerance_m = float(
            exploration_cfg.get("make_plan_tolerance_m", 0.20)
        )
        self.make_plan_endpoint_tolerance_m = float(
            exploration_cfg.get("make_plan_endpoint_tolerance_m", 0.60)
        )
        self.make_plan_fail_open = bool(
            exploration_cfg.get("make_plan_fail_open", True)
        )
        self.initial_local_goal_count = int(exploration_cfg.get("initial_local_goal_count", 3))
        self.latest_global_plan_pose_count = 0
        self.latest_global_plan_length_m = 0.0
        self.latest_global_plan_time = 0.0
        self.latest_global_plan_endpoint = None
        self.latest_global_plan_goal_distance_m = float("inf")
        self.latest_global_plan_matches_active_goal = False
        self.latest_local_plan_pose_count = 0
        self.latest_local_plan_length_m = 0.0
        self.latest_local_plan_time = 0.0
        self.local_plan_bad_since = 0.0
        self.last_goal_publish_time = 0.0
        self.last_status_key = ""
        self.seen_terminal_status_keys = set()
        self.last_move_base_feedback = None
        self.active_move_base_goal_id = ""
        self.active_goal_publish_ros_time = 0.0
        self.active_goal_publish_wall_time = 0.0
        self.sent_goal_count = 0
        self.rotation_replan_enabled = bool(exploration_cfg.get("rotation_replan_enabled", True))
        self.rotation_replan_window_sec = float(exploration_cfg.get("rotation_replan_window_sec", 10.0))
        self.rotation_replan_min_duration_sec = float(
            exploration_cfg.get("rotation_replan_min_duration_sec", 8.0)
        )
        self.rotation_replan_max_translation_m = float(
            exploration_cfg.get("rotation_replan_max_translation_m", 0.05)
        )
        self.rotation_replan_min_yaw_sum_rad = float(
            exploration_cfg.get("rotation_replan_min_yaw_sum_rad", 0.8)
        )
        self.rotation_replan_max_net_yaw_rad = float(
            exploration_cfg.get("rotation_replan_max_net_yaw_rad", 0.3)
        )
        self.rotation_replan_min_direction_changes = int(
            exploration_cfg.get("rotation_replan_min_direction_changes", 3)
        )
        self.rotation_replan_max_per_goal = int(exploration_cfg.get("rotation_replan_max_per_goal", 1))
        self.rotation_replan_cooldown_sec = float(exploration_cfg.get("rotation_replan_cooldown_sec", 20.0))
        self.rotation_replan_min_yaw_step_rad = float(
            exploration_cfg.get("rotation_replan_min_yaw_step_rad", 0.03)
        )
        self.rotation_replan_samples = []
        self.rotation_replan_goal_key = ""
        self.rotation_replan_count = 0
        self.rotation_replan_last_time = 0.0
        self.rotation_replan_last_metrics = {}

        core_config = FrontierConfig(
            free_max=int(frontier_cfg.get("free_max", 20)),
            occupied_min=int(frontier_cfg.get("occupied_min", 50)),
            hard_min_cluster_cells=int(frontier_cfg.get("hard_min_cluster_cells", 3)),
            min_cluster_cells=int(frontier_cfg.get("min_cluster_cells", 3)),
            connect_8=bool(frontier_cfg.get("connect_8", True)),
            candidate_top_k=int(frontier_cfg.get("candidate_top_k", 12)),
            sensor_range_m=float(frontier_cfg.get("sensor_range_m", 5.0)),
            unknown_component_radius_m=float(
                frontier_cfg.get(
                    "unknown_component_radius_m",
                    frontier_cfg.get("sensor_range_m", 5.0),
                )
            ),
            subgoal_search_radius_cells=int(frontier_cfg.get("subgoal_search_radius_cells", 8)),
            min_subgoal_distance_m=float(frontier_cfg.get("min_subgoal_distance_m", 0.75)),
            hard_min_subgoal_distance_m=float(
                frontier_cfg.get("hard_min_subgoal_distance_m", 0.50)
            ),
            target_frontier_offset_m=float(frontier_cfg.get("target_frontier_offset_m", 0.35)),
            use_voronoi_viewpoints=bool(frontier_cfg.get("use_voronoi_viewpoints", True)),
            min_viewpoint_frontier_distance_m=float(frontier_cfg.get("min_viewpoint_frontier_distance_m", 0.65)),
            max_viewpoint_frontier_distance_m=float(frontier_cfg.get("max_viewpoint_frontier_distance_m", 2.8)),
            clearance_weight=float(frontier_cfg.get("clearance_weight", 0.8)),
            frontier_offset_weight=float(frontier_cfg.get("frontier_offset_weight", 0.45)),
            local_horizon_m=float(frontier_cfg.get("local_horizon_m", 3.0)),
            local_horizon_penalty=float(frontier_cfg.get("local_horizon_penalty", 0.35)),
            far_cluster_penalty=float(frontier_cfg.get("far_cluster_penalty", 0.45)),
            far_cluster_penalty_saturation_m=float(frontier_cfg.get("far_cluster_penalty_saturation_m", 4.0)),
            min_obstacle_clearance_m=float(frontier_cfg.get("min_obstacle_clearance_m", 0.25)),
            max_clearance_check_m=float(frontier_cfg.get("max_clearance_check_m", 0.8)),
            robot_radius_m=float(frontier_cfg.get("robot_radius_m", 0.35)),
            footprint_safety_margin_m=float(frontier_cfg.get("footprint_safety_margin_m", 0.10)),
            require_footprint_free=bool(frontier_cfg.get("require_footprint_free", True)),
            footprint_unknown_is_free=bool(frontier_cfg.get("footprint_unknown_is_free", True)),
            turning_safety_margin_m=float(frontier_cfg.get("turning_safety_margin_m", 0.25)),
            require_turning_clearance=bool(frontier_cfg.get("require_turning_clearance", True)),
            information_weight=float(scoring_cfg.get("information_weight", 1.0)),
            distance_weight=float(scoring_cfg.get("distance_weight", 0.55)),
            semantic_weight=float(scoring_cfg.get("semantic_weight", 0.35)),
            llm_weight=float(scoring_cfg.get("llm_weight", 0.8)),
            revisit_penalty=float(scoring_cfg.get("revisit_penalty", 0.6)),
            failure_penalty=float(scoring_cfg.get("failure_penalty", 1.0)),
            receding_distance_weight=float(scoring_cfg.get("receding_distance_weight", 0.15)),
            previous_subgoal_weight=float(scoring_cfg.get("previous_subgoal_weight", 0.35)),
            continuity_cost_weight=float(scoring_cfg.get("continuity_cost_weight", 0.25)),
            continuity_cost_saturation_m=float(scoring_cfg.get("continuity_cost_saturation_m", 4.0)),
            near_frontier_relax_distance_m=float(frontier_cfg.get("near_frontier_relax_distance_m", 1.5)),
            relaxed_min_viewpoint_frontier_distance_m=float(
                frontier_cfg.get("relaxed_min_viewpoint_frontier_distance_m", 0.35)
            ),
            initial_local_radius_m=float(frontier_cfg.get("initial_local_radius_m", 2.2)),
            initial_backward_weight=float(frontier_cfg.get("initial_backward_weight", 0.35)),
        )
        state_config = ExplorerStateConfig(
            goal_reach_tolerance_m=float(exploration_cfg.get("goal_reach_tolerance_m", 0.35)),
            goal_timeout_sec=float(exploration_cfg.get("goal_timeout_sec", 90.0)),
            stall_timeout_sec=float(exploration_cfg.get("stall_timeout_sec", 30.0)),
            stall_distance_m=float(exploration_cfg.get("stall_distance_m", 0.15)),
            min_goal_lifetime_sec=float(exploration_cfg.get("min_goal_lifetime_sec", 8.0)),
            stall_yaw_progress_rad=float(exploration_cfg.get("stall_yaw_progress_rad", 0.20)),
            rotation_stall_timeout_sec=float(exploration_cfg.get("rotation_stall_timeout_sec", 45.0)),
            blacklist_duration_sec=float(exploration_cfg.get("blacklist_duration_sec", 25.0)),
            failed_cluster_retry_sec=float(exploration_cfg.get("failed_cluster_retry_sec", 120.0)),
            failed_cluster_max_failures=int(exploration_cfg.get("failed_cluster_max_failures", 3)),
            frontier_match_distance_m=float(exploration_cfg.get("frontier_match_distance_m", 1.0)),
            frontier_gone_confirm_ticks=self.frontier_gone_confirm_ticks,
            frontier_gone_min_goal_age_sec=self.frontier_gone_min_goal_age_sec,
            failed_point_soft_blacklist_sec=float(exploration_cfg.get("failed_point_soft_blacklist_sec", 10.0)),
            failed_point_blacklist_sec=float(exploration_cfg.get("failed_point_blacklist_sec", 180.0)),
            failed_point_blacklist_radius_m=float(exploration_cfg.get("failed_point_blacklist_radius_m", 1.25)),
            reached_point_blacklist_sec=float(exploration_cfg.get("reached_point_blacklist_sec", 90.0)),
            reached_point_blacklist_radius_m=float(exploration_cfg.get("reached_point_blacklist_radius_m", 0.75)),
            visit_viewpoint_once=bool(exploration_cfg.get("visit_viewpoint_once", False)),
            visited_viewpoint_radius_m=float(exploration_cfg.get("visited_viewpoint_radius_m", 0.50)),
            unreachable_frontier_radius_m=float(exploration_cfg.get("unreachable_frontier_radius_m", 1.0)),
        )

        self.core = FrontierExplorerCore(core_config)
        self.state = ExplorerState(state_config)
        self.value_fusion = ValueMapFusion()
        self.skill_api = ExplorationSkillApi(self)
        self.make_plan_client = rospy.ServiceProxy(self.make_plan_service, GetPlan)

        self.latest_grid_msg = None
        self.latest_grid = None
        self.robot_xy = None
        self.robot_yaw = None
        self.latest_clusters = []
        self.last_selected_cluster = None
        self.external_reserved_cluster = None
        self.external_reserved_command = None
        self.external_reservation_ack_cache = {}
        self.external_reservation_received_count = 0
        self.external_reservation_replay_count = 0
        self.external_reservation_last_command_id = ""
        self.external_reservation_last_cluster_id = ""
        self.external_reservation_last_status = ""
        self.external_reservation_last_detail = {}

        self.goal_pub = rospy.Publisher(self.topics.get("goal", "/move_base_simple/goal"), PoseStamped, queue_size=1)
        self.cancel_pub = rospy.Publisher(self.topics.get("move_base_cancel", "/move_base/cancel"), GoalID, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher(self.topics.get("cmd_vel", "/cmd_vel"), Twist, queue_size=1)
        self.status_pub = rospy.Publisher(self.topics.get("status", "/explore_py/status"), String, queue_size=1)
        self.behavior_feedback_pub = rospy.Publisher(
            self.topics.get("behavior_feedback", "/explore_py/behavior_feedback"),
            String,
            queue_size=4,
            latch=True,
        )
        self.frontier_pub = rospy.Publisher(self.topics.get("frontiers", "/explore_py/frontiers"), MarkerArray, queue_size=1)
        self.subgoal_pub = rospy.Publisher(
            self.topics.get("current_subgoal", "/explore_py/current_subgoal"), PointStamped, queue_size=1
        )

        rospy.Subscriber(self.topics.get("occupancy_grid", "/struct_mapping/occ_map"), OccupancyGrid, self.occupancy_callback, queue_size=1)
        rospy.Subscriber(self.topics.get("odom", "/odom"), Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(self.topics.get("move_base_status", "/move_base/status"), GoalStatusArray, self.move_base_status_callback, queue_size=10)
        rospy.Subscriber(self.topics.get("global_plan", "/move_base/GlobalPlanner/plan"), NavPath, self.global_plan_callback, queue_size=10)
        rospy.Subscriber(self.topics.get("local_plan", "/move_base/DWAPlannerROS/local_plan"), NavPath, self.local_plan_callback, queue_size=10)
        rospy.Subscriber(self.topics.get("llm_value_grid", "/explore_py/llm_value_grid"), OccupancyGrid, self.value_fusion.set_llm_value_grid, queue_size=1)
        rospy.Subscriber(self.topics.get("strategy_bias", "/explore_py/strategy_bias"), String, self.strategy_bias_callback, queue_size=1)
        rospy.Subscriber(self.topics.get("object_map", "/semantic_mapping/obj_map"), String, self.object_map_callback, queue_size=1)
        rospy.Subscriber(self.topics.get("scene_id_grid", "/semantic_mapping/scene_id_grid"), OccupancyGrid, self.value_fusion.set_scene_id_grid, queue_size=1)
        rospy.Subscriber(
            self.topics.get("scene_confidence_grid", "/semantic_mapping/scene_confidence_grid"),
            OccupancyGrid,
            self.value_fusion.set_scene_confidence_grid,
            queue_size=1,
        )
        rospy.Subscriber(self.topics.get("unified_graph", "/semantic_mapping/unified_graph"), String, self.unified_graph_callback, queue_size=1)
        rospy.Subscriber(
            self.topics.get("navigation_hints", "/semantic_mapping/navigation_hints"),
            String,
            self.navigation_hints_callback,
            queue_size=1,
        )
        rospy.Subscriber(self.topics.get("reset", "/explore_py/reset"), Empty, self.reset_callback, queue_size=1)
        rospy.Subscriber(
            self.topics.get("behavior_command", "/explore_py/command"),
            String,
            self.behavior_command_callback,
            queue_size=4,
        )

        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.tick_rate_hz, 1e-3)), self.tick)
        self.initial_spin_cmd_timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.initial_spin_cmd_rate_hz, 1e-3)),
            self._initial_spin_cmd_timer_callback,
        )
        rospy.loginfo("[explore_py] initialized: occ=%s goal=%s status=%s",
                      self.topics.get("occupancy_grid", "/struct_mapping/occ_map"),
                      self.topics.get("goal", "/move_base_simple/goal"),
                      self.topics.get("move_base_status", "/move_base/status"))

    def reset_callback(self, _msg):
        self._cancel_move_base_goal("external_reset")
        self.state = ExplorerState(self.state.config)
        self.value_fusion = ValueMapFusion()
        self.latest_grid_msg = None
        self.latest_grid = None
        self.robot_xy = None
        self.robot_yaw = None
        self.latest_clusters = []
        self.last_selected_cluster = None
        self.external_reserved_cluster = None
        self.external_reserved_command = None
        self.external_reservation_ack_cache.clear()
        self.external_reservation_received_count = 0
        self.external_reservation_replay_count = 0
        self.external_reservation_last_command_id = ""
        self.external_reservation_last_cluster_id = ""
        self.external_reservation_last_status = ""
        self.external_reservation_last_detail = {}
        self.latest_global_plan_pose_count = 0
        self.latest_global_plan_length_m = 0.0
        self.latest_global_plan_time = 0.0
        self.latest_global_plan_endpoint = None
        self.latest_global_plan_goal_distance_m = float("inf")
        self.latest_global_plan_matches_active_goal = False
        self.latest_local_plan_pose_count = 0
        self.latest_local_plan_length_m = 0.0
        self.latest_local_plan_time = 0.0
        self.local_plan_bad_since = 0.0
        self.last_goal_publish_time = 0.0
        self.last_status_key = ""
        self.seen_terminal_status_keys.clear()
        self.last_move_base_feedback = None
        self.active_move_base_goal_id = ""
        self.active_goal_publish_ros_time = 0.0
        self.active_goal_publish_wall_time = 0.0
        self.sent_goal_count = 0
        self._reset_rotation_replan_tracking()
        self.initial_spin_done = not self.initial_spin_enabled
        self.initial_spin_active = False
        self.initial_spin_start_time = 0.0
        self.initial_spin_done_time = 0.0
        self.initial_spin_last_yaw = None
        self.initial_spin_accumulated_yaw = 0.0
        self.initial_spin_reason = "disabled" if self.initial_spin_done else "pending"
        marker = Marker()
        marker.action = Marker.DELETEALL
        self.frontier_pub.publish(MarkerArray(markers=[marker]))
        rospy.logwarn("[explore_py] exploration state reset for a new scene")

    def occupancy_callback(self, msg):
        self.latest_grid_msg = msg
        self.latest_grid = self._convert_grid(msg)

    def odom_callback(self, msg):
        self.robot_xy = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))
        self.robot_yaw = self._yaw_from_quaternion(msg.pose.pose.orientation)

    def strategy_bias_callback(self, msg):
        self.value_fusion.set_strategy_bias_json(msg.data)

    def object_map_callback(self, msg):
        self.value_fusion.set_object_map_json(msg.data)

    def unified_graph_callback(self, msg):
        self.value_fusion.set_unified_graph_json(msg.data)

    def navigation_hints_callback(self, msg):
        self.value_fusion.set_navigation_hints_json(msg.data)

    def behavior_command_callback(self, msg):
        try:
            command = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        action = str(command.get("action") or "")
        if action == "reserve_frontier":
            try:
                self._reserve_external_frontier(command)
            except Exception as error:
                command_id = str(command.get("command_id") or "")
                detail = {"reason": "reservation_exception", "error": str(error)}
                self.external_reservation_last_command_id = command_id
                self.external_reservation_last_cluster_id = str(command.get("cluster_id") or "")
                self.external_reservation_last_status = "FAILED"
                self.external_reservation_last_detail = detail
                self.external_reservation_ack_cache[command_id] = {
                    "status": "FAILED",
                    "success": False,
                    "detail": detail,
                }
                rospy.logerr("[explore_py] external frontier reservation failed: command=%s error=%s", command_id, error)
                self._publish_behavior_feedback(command, "FAILED", False, detail)
        elif action == "finalize_frontier":
            self._finalize_external_frontier(command)

    def global_plan_callback(self, msg: NavPath):
        self.latest_global_plan_pose_count = len(msg.poses)
        self.latest_global_plan_length_m = self._path_length(msg)
        self.latest_global_plan_time = time.time()
        self.latest_global_plan_endpoint = None
        self.latest_global_plan_goal_distance_m = float("inf")
        self.latest_global_plan_matches_active_goal = False
        if msg.poses:
            endpoint = msg.poses[-1].pose.position
            self.latest_global_plan_endpoint = (float(endpoint.x), float(endpoint.y))
            if self.state.active_goal is not None:
                goal = self.state.active_goal.point
                self.latest_global_plan_goal_distance_m = math.hypot(
                    self.latest_global_plan_endpoint[0] - goal[0],
                    self.latest_global_plan_endpoint[1] - goal[1],
                )
                self.latest_global_plan_matches_active_goal = (
                    self.latest_global_plan_goal_distance_m <= self.global_plan_goal_tolerance_m
                )

    def local_plan_callback(self, msg: NavPath):
        self.latest_local_plan_pose_count = len(msg.poses)
        self.latest_local_plan_length_m = self._path_length(msg)
        self.latest_local_plan_time = time.time()

    @staticmethod
    def _path_length(msg: NavPath) -> float:
        total = 0.0
        last = None
        for pose_stamped in getattr(msg, "poses", []) or []:
            position = pose_stamped.pose.position
            current = (float(position.x), float(position.y))
            if last is not None:
                total += math.hypot(current[0] - last[0], current[1] - last[1])
            last = current
        return total

    def move_base_status_callback(self, msg):
        if self.external_behavior_control:
            return
        statuses = list(getattr(msg, "status_list", []) or [])
        has_active_goal = self.state.active_goal is not None

        if has_active_goal:
            for status in statuses:
                code = int(status.status)
                if code not in (0, 1):
                    continue
                goal_id = getattr(status.goal_id, "id", "")
                goal_stamp = self._goal_id_stamp(status)
                if self._status_belongs_to_active_goal(goal_id, goal_stamp):
                    if goal_id and goal_id != self.active_move_base_goal_id:
                        rospy.loginfo("[explore_py] bound move_base goal id: %s", goal_id)
                    self.active_move_base_goal_id = goal_id

        for status in statuses:
            code = int(status.status)
            if code not in TERMINAL_FAILURE and code not in TERMINAL_SUCCESS:
                continue
            goal_id = getattr(status.goal_id, "id", "")
            goal_stamp = self._goal_id_stamp(status)
            feedback = {
                "goal_id": goal_id,
                "status": code,
                "status_name": status_name(code),
                "text": getattr(status, "text", ""),
                "stamp": msg.header.stamp.to_sec() if msg.header.stamp else 0.0,
                "goal_stamp": goal_stamp,
                "is_terminal_failure": code in TERMINAL_FAILURE,
                "is_terminal_success": code in TERMINAL_SUCCESS,
            }
            self.last_move_base_feedback = feedback
            status_key = f"{goal_id}:{code}:{getattr(status, 'text', '')}"
            if status_key in self.seen_terminal_status_keys:
                continue
            self.seen_terminal_status_keys.add(status_key)
            self.last_status_key = status_key
            if not has_active_goal:
                continue
            if not self._status_belongs_to_active_goal(goal_id, goal_stamp):
                rospy.logdebug(
                    "[explore_py] ignored terminal status for stale move_base goal id=%s active=%s status=%s",
                    goal_id,
                    self.active_move_base_goal_id,
                    status_name(code),
                )
                continue
            if code in TERMINAL_FAILURE:
                rospy.logwarn("[explore_py] move_base reported %s, replanning next tick", status_name(code))
                self.state.mark_active_failed(f"move_base_{status_name(code).lower()}", source="move_base")
                self.active_move_base_goal_id = ""
                return
            if code in TERMINAL_SUCCESS and self._active_goal_distance() <= self.state.config.goal_reach_tolerance_m * 1.5:
                if self._active_goal_has_frontier():
                    self.state.mark_active_frontier_unreachable()
                else:
                    self.state.mark_active_reached()
                self.active_move_base_goal_id = ""
                return

    def tick(self, _event):
        if self.latest_grid is None or self.robot_xy is None:
            self._publish_status()
            return

        if self._should_run_initial_spin():
            self.compute_next_subgoal(force=False, publish_selection=True)
            self._tick_initial_spin()
            self._publish_frontiers()
            self._publish_status()
            return

        if self.external_behavior_control:
            self.compute_next_subgoal(force=False, publish_selection=False)
            self._tick_external_navigation_progress()
            self._publish_frontiers()
            self._publish_status()
            return

        active_goal_before_update = self.state.active_goal is not None
        if self.state.active_goal is not None:
            if not self.core.is_free_world(self.latest_grid, self.state.active_goal.point):
                self.state.fail_active_if_goal_not_free(False)
            else:
                has_frontier = self._active_goal_has_frontier()
                if self._maybe_replan_rotation_oscillation():
                    self._publish_frontiers()
                    self._publish_status()
                    return
                progress = self.state.update_goal_progress(self.robot_xy, robot_yaw=self.robot_yaw)
                if progress == SUBGOAL_REACHED:
                    if has_frontier:
                        self.state.mark_active_frontier_unreachable()
                    else:
                        self.state.mark_active_reached()
                elif self.state.active_goal is not None:
                    self._fail_if_global_plan_not_current_goal()
                if self.state.active_goal is not None:
                    self._fail_if_local_plan_missing()
                if self.state.active_goal is not None:
                    self.state.mark_active_covered_if_frontier_gone(
                        has_frontier,
                        confirm_ticks=self.frontier_gone_confirm_ticks,
                        min_goal_age_sec=self.frontier_gone_min_goal_age_sec,
                    )
        if active_goal_before_update and self.state.active_goal is None:
            self._cancel_move_base_goal(self.state.last_event or "explorer_closed_goal")
            self.active_move_base_goal_id = ""

        if self.state.active_goal is None:
            cluster = self.compute_next_subgoal(force=True)
            if cluster is not None:
                self._send_goal(cluster)
        else:
            now = time.time()
            if self.goal_republish_interval_sec > 0.0 and now - self.last_goal_publish_time >= self.goal_republish_interval_sec:
                self._publish_active_goal()

        self._publish_frontiers()
        self._publish_status()

    def _active_goal_has_frontier(self) -> bool:
        goal = self.state.active_goal
        if goal is None:
            return False
        if self.latest_grid is None:
            # Without a current map, navigation success must not erase exploration work.
            return True
        return self.core.has_frontier_near(
            self.latest_grid,
            goal.frontier_point,
            self.state.config.frontier_match_distance_m,
            min_cells=self.active_goal_frontier_min_cells,
        )

    def _fail_if_local_plan_missing(self):
        if not self.local_plan_watchdog_enabled or self.state.active_goal is None:
            self.local_plan_bad_since = 0.0
            return
        now = time.time()
        goal_age = now - self.state.active_goal.sent_at
        if goal_age < self.state.config.min_goal_lifetime_sec:
            self.local_plan_bad_since = 0.0
            return
        global_fresh = now - self.latest_global_plan_time <= self.plan_freshness_sec
        local_fresh = now - self.latest_local_plan_time <= self.plan_freshness_sec
        global_available = global_fresh and self.latest_global_plan_pose_count >= self.global_plan_min_poses
        local_available = (
            local_fresh
            and self.latest_local_plan_pose_count >= self.local_plan_min_poses
            and self.latest_local_plan_length_m >= self.local_plan_min_length_m
        )
        if not global_available or local_available:
            self.local_plan_bad_since = 0.0
            return
        goal = self.state.active_goal
        translation_stalled = (
            goal is not None
            and now - goal.last_progress_at >= self.local_plan_watchdog_sec
        )
        rotation_dominant = (
            goal is not None
            and goal.last_yaw_progress_at >= goal.last_progress_at
        )
        if not translation_stalled or not rotation_dominant:
            self.local_plan_bad_since = 0.0
            return
        if self.local_plan_bad_since <= 0.0:
            self.local_plan_bad_since = now
            return
        bad_duration = now - self.local_plan_bad_since
        if bad_duration >= self.local_plan_watchdog_sec:
            rospy.logwarn(
                "[explore_py] local plan watchdog failed active goal after rotation-dominant no-translation: global_poses=%d local_poses=%d local_len=%.2fm bad_duration=%.1fs",
                self.latest_global_plan_pose_count,
                self.latest_local_plan_pose_count,
                self.latest_local_plan_length_m,
                bad_duration,
            )
            self.state.mark_active_failed(
                "local_plan_degenerate_no_translation", source="explorer"
            )
            self.local_plan_bad_since = 0.0

    def _tick_external_navigation_progress(self):
        """Keep an externally reserved frontier alive until the executor finalizes it.

        In semantic-control mode the executor owns make_plan, rear-goal rotation,
        move_base completion, and final yaw alignment.  Applying the legacy
        explorer position/plan watchdogs here races that state machine: a goal
        can be erased while the executor is still planning or rotating.
        """
        if self.external_reserved_command is None or self.state.active_goal is None:
            return

    def _fail_if_global_plan_not_current_goal(self):
        if not self.global_plan_current_goal_check_enabled or self.state.active_goal is None:
            return
        now = time.time()
        # A forced same-goal replan has a new publication time even though the
        # exploration goal keeps its original lifetime and timeout budget.
        goal_age = now - max(self.state.active_goal.sent_at, self.active_goal_publish_wall_time)
        grace_sec = self.global_plan_current_goal_grace_sec
        if self.external_behavior_control and self.external_reserved_command is not None:
            grace_sec = max(grace_sec, self.external_navigation_plan_grace_sec)
        if goal_age < grace_sec:
            return
        fresh_after_goal = self.latest_global_plan_time >= self.active_goal_publish_wall_time
        plan_available = fresh_after_goal and self.latest_global_plan_pose_count >= self.global_plan_min_poses
        if plan_available and self.latest_global_plan_matches_active_goal:
            return
        reason = "global_plan_not_for_current_goal" if plan_available else "global_plan_missing_current_goal"
        rospy.logwarn(
            "[explore_py] current goal has no matching global plan: reason=%s poses=%d endpoint=%s goal_dist=%.2f age=%.1fs",
            reason,
            self.latest_global_plan_pose_count,
            self.latest_global_plan_endpoint,
            self.latest_global_plan_goal_distance_m,
            goal_age,
        )
        self.state.mark_active_failed(reason, source="explorer")

    def compute_next_subgoal(self, force=False, publish_selection=True):
        if self.latest_grid is None or self.robot_xy is None:
            return None
        clusters = self.core.extract_frontier_clusters(
            self.latest_grid,
            self.robot_xy,
            value_provider=self.value_fusion,
            state=self.state,
        )
        self.state.update_seen_clusters(clusters)
        self.latest_clusters = clusters
        if not clusters:
            if publish_selection:
                self.last_selected_cluster = None
            return None
        if self.sent_goal_count < self.initial_local_goal_count:
            cluster = self.core.select_initial_local_cluster(clusters, self.robot_xy, robot_yaw=self.robot_yaw)
        else:
            ranked = self.core.rank_clusters(clusters, self.robot_xy, state=self.state)
            cluster = ranked[0] if ranked else None
        if publish_selection:
            self.last_selected_cluster = cluster
        return cluster

    def build_status_payload(self):
        payload = {
            "ready": self.latest_grid is not None and self.robot_xy is not None,
            "external_behavior_control": self.external_behavior_control,
            "robot_xy": list(self.robot_xy) if self.robot_xy is not None else None,
            "map_resolution": (
                float(self.latest_grid.spec.resolution)
                if self.latest_grid is not None
                else None
            ),
            "frontier_count": len(self.latest_clusters),
            "frontier_debug": self.core.last_debug_stats,
            "frontier_clusters": [
                self._cluster_to_dict(cluster, grid=self.latest_grid, include_cells=True)
                for cluster in self.latest_clusters
            ],
            "selected_cluster": self._cluster_to_dict(self.last_selected_cluster),
            "state": self.state.summary(),
            "semantic_inputs": {
                "object_count": len(self.value_fusion.object_map),
                "navigation_hint_count": len(self.value_fusion.navigation_hints),
                "has_llm_value_grid": self.value_fusion.llm_value_grid is not None,
                "strategy_bias": self.value_fusion.strategy_bias,
            },
            "initial_spin": {
                "enabled": self.initial_spin_enabled,
                "done": self.initial_spin_done,
                "active": self.initial_spin_active,
                "accumulated_yaw_rad": self.initial_spin_accumulated_yaw,
                "target_yaw_rad": self.initial_spin_angle_rad,
                "cmd_rate_hz": self.initial_spin_cmd_rate_hz,
                "reason": self.initial_spin_reason,
            },
            "plan_watchdog": {
                "enabled": self.local_plan_watchdog_enabled,
                "global_plan_poses": self.latest_global_plan_pose_count,
                "global_plan_length_m": self.latest_global_plan_length_m,
                "global_plan_endpoint": self.latest_global_plan_endpoint,
                "global_plan_goal_distance_m": self._finite_or_none(self.latest_global_plan_goal_distance_m),
                "global_plan_matches_active_goal": self.latest_global_plan_matches_active_goal,
                "local_plan_poses": self.latest_local_plan_pose_count,
                "local_plan_length_m": self.latest_local_plan_length_m,
                "local_plan_bad_since": self.local_plan_bad_since,
            },
            "rotation_replan": {
                "enabled": self.rotation_replan_enabled,
                "count": self.rotation_replan_count,
                "max_per_goal": self.rotation_replan_max_per_goal,
                "sample_count": len(self.rotation_replan_samples),
                "last_metrics": self.rotation_replan_last_metrics,
            },
            "external_reservation": {
                "received_count": self.external_reservation_received_count,
                "replay_count": self.external_reservation_replay_count,
                "cache_size": len(self.external_reservation_ack_cache),
                "last_command_id": self.external_reservation_last_command_id,
                "last_cluster_id": self.external_reservation_last_cluster_id,
                "last_status": self.external_reservation_last_status,
                "last_detail": dict(self.external_reservation_last_detail),
            },
        }
        if self.state.active_goal is not None:
            payload["active_goal_distance"] = self._active_goal_distance()
        if self.last_move_base_feedback is not None:
            payload["move_base_feedback"] = self.last_move_base_feedback
        return payload

    def _reserve_external_frontier(self, command):
        if not self.external_behavior_control:
            self._publish_behavior_feedback(
                command,
                "REJECTED",
                False,
                {"reason": "external_behavior_control_disabled"},
            )
            return
        command_id = str(command.get("command_id") or "")
        cluster_id = str(command.get("cluster_id") or "")
        self.external_reservation_received_count += 1
        self.external_reservation_last_command_id = command_id
        self.external_reservation_last_cluster_id = cluster_id
        cached_ack = self.external_reservation_ack_cache.get(command_id)
        if cached_ack is not None:
            self.external_reservation_replay_count += 1
            self.external_reservation_last_status = str(cached_ack["status"])
            self.external_reservation_last_detail = dict(cached_ack["detail"])
            rospy.loginfo("[explore_py] replaying reservation ACK: command=%s status=%s", command_id, cached_ack["status"])
            self._publish_behavior_feedback(
                command,
                cached_ack["status"],
                cached_ack["success"],
                cached_ack["detail"],
            )
            return
        cluster = next(
            (
                candidate
                for candidate in self.latest_clusters
                if str(candidate.cluster_id) == cluster_id
            ),
            None,
        )
        if cluster is None:
            detail = {"reason": "frontier_not_available", "cluster_id": cluster_id}
            self.external_reservation_ack_cache[command_id] = {
                "status": "FAILED",
                "success": False,
                "detail": detail,
            }
            self.external_reservation_last_status = "FAILED"
            self.external_reservation_last_detail = detail
            rospy.logwarn("[explore_py] reservation rejected: command=%s cluster=%s latest_clusters=%d", command_id, cluster_id, len(self.latest_clusters))
            self._publish_behavior_feedback(
                command,
                "FAILED",
                False,
                detail,
            )
            return
        requested_goal = list(command.get("goal_xyyaw") or [])
        current_goal = [
            float(cluster.subgoal_world[0]),
            float(cluster.subgoal_world[1]),
            float(cluster.subgoal_yaw),
        ]
        reservation_goal_drift_m = 0.0
        try:
            requested_goal_valid = len(requested_goal) >= 2 and all(
                math.isfinite(float(value)) for value in requested_goal[:3]
            )
        except (TypeError, ValueError):
            requested_goal_valid = False
        if requested_goal_valid:
            requested_yaw = (
                float(requested_goal[2])
                if len(requested_goal) > 2
                else float(cluster.subgoal_yaw)
            )
            reservation_goal_drift_m = math.hypot(
                float(requested_goal[0]) - current_goal[0],
                float(requested_goal[1]) - current_goal[1],
            )
            cluster = replace(
                cluster,
                subgoal_world=(
                    float(requested_goal[0]),
                    float(requested_goal[1]),
                ),
                subgoal_yaw=requested_yaw,
                distance_to_robot=(
                    math.hypot(
                        float(requested_goal[0]) - self.robot_xy[0],
                        float(requested_goal[1]) - self.robot_xy[1],
                    )
                    if self.robot_xy is not None
                    else cluster.distance_to_robot
                ),
            )
        self.external_reserved_cluster = cluster
        self.external_reserved_command = dict(command)
        self.last_selected_cluster = cluster
        self.active_goal_publish_wall_time = time.time()
        if self.robot_xy is not None:
            self.state.start_goal(
                cluster,
                self.robot_xy,
                robot_yaw=self.robot_yaw,
                goal_id=str(cluster.cluster_id),
            )
        point_msg = PointStamped()
        point_msg.header.stamp = rospy.Time.now()
        point_msg.header.frame_id = self.map_frame
        point_msg.point = Point(cluster.subgoal_world[0], cluster.subgoal_world[1], 0.0)
        self.subgoal_pub.publish(point_msg)
        detail = {
            "cluster_id": cluster.cluster_id,
            "goal_xyyaw": [
                float(cluster.subgoal_world[0]),
                float(cluster.subgoal_world[1]),
                float(cluster.subgoal_yaw),
            ],
            "frame_id": self.map_frame,
            "candidate_sequence": int(command.get("candidate_sequence", 0) or 0),
            "graph_revision": int(command.get("graph_revision", 0) or 0),
            "current_cluster_goal_xyyaw": current_goal,
            "reservation_goal_drift_m": reservation_goal_drift_m,
            "selected_goal_preserved": requested_goal_valid,
        }
        self.external_reservation_ack_cache[command_id] = {
            "status": "READY",
            "success": None,
            "detail": detail,
        }
        self.external_reservation_last_status = "READY"
        self.external_reservation_last_detail = detail
        rospy.loginfo("[explore_py] reservation ready: command=%s cluster=%s", command_id, cluster_id)
        self._publish_behavior_feedback(command, "READY", None, detail)

    def _finalize_external_frontier(self, command):
        cluster = self.external_reserved_cluster
        success = bool(command.get("success"))
        detail = dict(command.get("detail") or {})
        canceled = str(detail.get("reason") or "") == "preempted_by_target"
        if cluster is not None and self.robot_xy is not None:
            if self.state.active_goal is None:
                self.state.start_goal(
                    cluster,
                    self.robot_xy,
                    robot_yaw=self.robot_yaw,
                    goal_id=str(cluster.cluster_id),
                )
            if canceled:
                self.state.clear_active_goal(
                    "subgoal_canceled",
                    event="preempted_by_target",
                )
            elif success:
                has_frontier = self.core.has_frontier_near(
                    self.latest_grid,
                    cluster.centroid_world,
                    self.state.config.frontier_match_distance_m,
                    min_cells=self.active_goal_frontier_min_cells,
                ) if self.latest_grid is not None else False
                if has_frontier:
                    self.state.mark_active_frontier_unreachable()
                else:
                    self.state.mark_active_reached()
            else:
                self.state.mark_active_failed(
                    str(detail.get("reason") or "semantic_executor_navigation_failed"),
                    source="semantic_executor",
                )
        self.external_reserved_cluster = None
        self.external_reserved_command = None
        self._publish_behavior_feedback(
            command,
            "CANCELED" if canceled else "SUCCEEDED" if success else "FAILED",
            success,
            {
                "cluster_id": command.get("cluster_id", ""),
                **detail,
            },
        )

    def _publish_behavior_feedback(self, command, status, success, detail):
        payload = {
            "command_id": command.get("command_id", ""),
            "decision_id": command.get("decision_id", ""),
            "candidate_id": command.get("candidate_id", ""),
            "cluster_id": command.get("cluster_id", ""),
            "status": status,
            "success": success,
            "detail": dict(detail or {}),
            "timestamp": time.time(),
        }
        self.behavior_feedback_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )

    def _send_goal(self, cluster):
        if not self._preflight_cluster_plan(cluster):
            self.last_selected_cluster = cluster
            self.state.start_goal(
                cluster,
                self.robot_xy,
                robot_yaw=self.robot_yaw,
                goal_id=cluster.cluster_id,
            )
            self.state.mark_active_failed("make_plan_unreachable", source="make_plan")
            self.state.blacklist_cluster(cluster.cluster_id)
            self._publish_status()
            rospy.logwarn(
                "[explore_py] rejected unreachable subgoal before publish: cluster=%s point=(%.2f, %.2f)",
                cluster.cluster_id,
                cluster.subgoal_world[0],
                cluster.subgoal_world[1],
            )
            return False
        self._reset_rotation_replan_tracking()
        self.local_plan_bad_since = 0.0
        self.latest_global_plan_pose_count = 0
        self.latest_global_plan_length_m = 0.0
        self.latest_global_plan_time = 0.0
        self.latest_global_plan_endpoint = None
        self.latest_global_plan_goal_distance_m = float("inf")
        self.latest_global_plan_matches_active_goal = False
        self.latest_local_plan_pose_count = 0
        self.latest_local_plan_length_m = 0.0
        self.latest_local_plan_time = 0.0
        self.active_move_base_goal_id = ""
        self.active_goal_publish_ros_time = 0.0
        self.active_goal_publish_wall_time = 0.0
        self.last_selected_cluster = cluster
        self.state.start_goal(cluster, self.robot_xy, robot_yaw=self.robot_yaw, goal_id=cluster.cluster_id)
        self.sent_goal_count += 1
        self._publish_status()
        self._publish_active_goal()
        rospy.loginfo("[explore_py] sent subgoal cluster=%s point=(%.2f, %.2f, yaw=%.2f) score=%.3f reason=%s",
                      cluster.cluster_id,
                      cluster.subgoal_world[0],
                      cluster.subgoal_world[1],
                      cluster.subgoal_yaw,
                      cluster.score,
                      json.dumps(cluster.score_terms, sort_keys=True))
        return True

    def _preflight_cluster_plan(self, cluster) -> bool:
        if not self.make_plan_preflight_enabled:
            return True
        if self.robot_xy is None:
            return self.make_plan_fail_open
        stamp = rospy.Time.now()
        start = PoseStamped()
        start.header.frame_id = self.map_frame
        start.header.stamp = stamp
        start.pose.position.x = float(self.robot_xy[0])
        start.pose.position.y = float(self.robot_xy[1])
        start_yaw = 0.0 if self.robot_yaw is None else float(self.robot_yaw)
        start.pose.orientation.z = math.sin(0.5 * start_yaw)
        start.pose.orientation.w = math.cos(0.5 * start_yaw)
        goal = PoseStamped()
        goal.header.frame_id = self.map_frame
        goal.header.stamp = stamp
        goal.pose.position.x = float(cluster.subgoal_world[0])
        goal.pose.position.y = float(cluster.subgoal_world[1])
        goal.pose.orientation.z = math.sin(0.5 * float(cluster.subgoal_yaw))
        goal.pose.orientation.w = math.cos(0.5 * float(cluster.subgoal_yaw))
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
                "[explore_py] make_plan preflight unavailable: %s",
                exc,
            )
            return self.make_plan_fail_open
        poses = list(response.plan.poses or [])
        if not poses:
            return False
        endpoint = poses[-1].pose.position
        return math.hypot(
            float(endpoint.x) - float(cluster.subgoal_world[0]),
            float(endpoint.y) - float(cluster.subgoal_world[1]),
        ) <= max(self.make_plan_endpoint_tolerance_m, self.make_plan_tolerance_m)

    def _reset_rotation_replan_tracking(self):
        self.rotation_replan_samples = []
        self.rotation_replan_goal_key = ""
        self.rotation_replan_count = 0
        self.rotation_replan_last_time = 0.0
        self.rotation_replan_last_metrics = {}

    def _maybe_replan_rotation_oscillation(self) -> bool:
        goal = self.state.active_goal
        if (
            not self.rotation_replan_enabled
            or goal is None
            or self.robot_xy is None
            or self.robot_yaw is None
        ):
            return False

        goal_key = f"{goal.cluster_id}:{goal.sent_at:.6f}"
        if goal_key != self.rotation_replan_goal_key:
            self.rotation_replan_goal_key = goal_key
            self.rotation_replan_samples = []
            self.rotation_replan_count = 0
            self.rotation_replan_last_time = 0.0
            self.rotation_replan_last_metrics = {}

        now = time.time()
        self.rotation_replan_samples.append(
            (now, float(self.robot_xy[0]), float(self.robot_xy[1]), float(self.robot_yaw))
        )
        cutoff = now - max(1.0, self.rotation_replan_window_sec)
        self.rotation_replan_samples = [
            sample for sample in self.rotation_replan_samples if sample[0] >= cutoff
        ]
        if len(self.rotation_replan_samples) < 3:
            return False

        first = self.rotation_replan_samples[0]
        last = self.rotation_replan_samples[-1]
        duration = last[0] - first[0]
        max_translation = max(
            math.hypot(sample[1] - first[1], sample[2] - first[2])
            for sample in self.rotation_replan_samples
        )
        yaw_deltas = [
            self._signed_angle_diff(current[3], previous[3])
            for previous, current in zip(
                self.rotation_replan_samples,
                self.rotation_replan_samples[1:],
            )
        ]
        yaw_sum = sum(abs(delta) for delta in yaw_deltas)
        net_yaw = abs(self._signed_angle_diff(last[3], first[3]))
        signs = []
        for delta in yaw_deltas:
            if abs(delta) < self.rotation_replan_min_yaw_step_rad:
                continue
            signs.append(1 if delta > 0.0 else -1)
        direction_changes = sum(
            current != previous for previous, current in zip(signs, signs[1:])
        )
        metrics = {
            "duration_sec": duration,
            "max_translation_m": max_translation,
            "yaw_sum_rad": yaw_sum,
            "net_yaw_rad": net_yaw,
            "direction_changes": direction_changes,
        }
        self.rotation_replan_last_metrics = metrics

        oscillating = (
            duration >= self.rotation_replan_min_duration_sec
            and max_translation <= self.rotation_replan_max_translation_m
            and yaw_sum >= self.rotation_replan_min_yaw_sum_rad
            and net_yaw <= self.rotation_replan_max_net_yaw_rad
            and direction_changes >= self.rotation_replan_min_direction_changes
        )
        if not oscillating:
            return False
        if self.rotation_replan_count >= max(0, self.rotation_replan_max_per_goal):
            return False
        if now - self.rotation_replan_last_time < max(0.0, self.rotation_replan_cooldown_sec):
            return False

        self.rotation_replan_count += 1
        self.rotation_replan_last_time = now
        self.rotation_replan_samples = [last]
        self.local_plan_bad_since = 0.0
        self.latest_global_plan_pose_count = 0
        self.latest_global_plan_length_m = 0.0
        self.latest_global_plan_time = 0.0
        self.latest_global_plan_endpoint = None
        self.latest_global_plan_goal_distance_m = float("inf")
        self.latest_global_plan_matches_active_goal = False
        self.latest_local_plan_pose_count = 0
        self.latest_local_plan_length_m = 0.0
        self.latest_local_plan_time = 0.0
        self.active_move_base_goal_id = ""
        goal.last_progress_at = now
        goal.last_yaw_progress_at = now
        goal.last_robot_xy = self.robot_xy
        goal.last_robot_yaw = self.robot_yaw
        self.state.last_event = "rotation_oscillation_global_replan"
        self.state.last_replan_reason = "rotation_oscillation"
        self.state.last_replan_source = "explorer"
        self._publish_zero_cmd_vel()
        self._publish_active_goal()
        rospy.logwarn(
            "[explore_py] rotation oscillation triggered one global replan: "
            "duration=%.1fs move=%.3fm yaw_sum=%.2f net_yaw=%.2f direction_changes=%d",
            duration,
            max_translation,
            yaw_sum,
            net_yaw,
            direction_changes,
        )
        return True

    def _publish_active_goal(self):
        goal = self.state.active_goal
        if goal is None:
            return
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = goal.point[0]
        msg.pose.position.y = goal.point[1]
        msg.pose.position.z = 0.0
        qz, qw = self._quaternion_z_w_from_yaw(goal.yaw)
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.active_goal_publish_ros_time = msg.header.stamp.to_sec()
        self.active_goal_publish_wall_time = time.time()
        self.goal_pub.publish(msg)

        point_msg = PointStamped()
        point_msg.header = msg.header
        point_msg.point = Point(goal.point[0], goal.point[1], 0.0)
        self.subgoal_pub.publish(point_msg)
        goal.status = SUBGOAL_WAITING
        self.last_goal_publish_time = time.time()

    def _publish_status(self):
        self.status_pub.publish(String(data=json.dumps(self.build_status_payload(), ensure_ascii=False, sort_keys=True)))

    def _publish_frontiers(self):
        markers = MarkerArray()
        now = rospy.Time.now()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        scores = [float(cluster.score) for cluster in self.latest_clusters]
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 1.0
        score_span = max(max_score - min_score, 1e-6)
        for index, cluster in enumerate(self.latest_clusters):
            normalized_score = (float(cluster.score) - min_score) / score_span
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = now
            marker.ns = "voronoi_viewpoints"
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = cluster.subgoal_world[0]
            marker.pose.position.y = cluster.subgoal_world[1]
            marker.pose.position.z = 0.05
            marker.pose.orientation.w = 1.0
            scale = max(0.12, min(0.8, math.sqrt(len(cluster.cells)) * 0.06))
            marker.scale.x = marker.scale.y = marker.scale.z = scale
            marker.color.r = 1.0 - 0.85 * normalized_score
            marker.color.g = 0.25 + 0.70 * normalized_score
            marker.color.b = 0.15
            marker.color.a = 0.90
            markers.markers.append(marker)

            link = Marker()
            link.header = marker.header
            link.ns = "voronoi_frontier_links"
            link.id = 2000 + index
            link.type = Marker.LINE_LIST
            link.action = Marker.ADD
            link.pose.orientation.w = 1.0
            link.scale.x = 0.035
            link.color.r = 0.95
            link.color.g = 0.80
            link.color.b = 0.15
            link.color.a = 0.75
            link.points.append(Point(cluster.subgoal_world[0], cluster.subgoal_world[1], 0.10))
            link.points.append(Point(cluster.centroid_world[0], cluster.centroid_world[1], 0.10))
            markers.markers.append(link)

            if self.last_selected_cluster is not None and cluster.cluster_id == self.last_selected_cluster.cluster_id:
                frontier_points = Marker()
                frontier_points.header = marker.header
                frontier_points.ns = "selected_frontier_cells"
                frontier_points.id = 5000 + index
                frontier_points.type = Marker.POINTS
                frontier_points.action = Marker.ADD
                frontier_points.pose.orientation.w = 1.0
                frontier_points.scale.x = 0.08
                frontier_points.scale.y = 0.08
                frontier_points.color.r = 1.0
                frontier_points.color.g = 0.55
                frontier_points.color.b = 0.05
                frontier_points.color.a = 0.95
                for cell_x, cell_y in cluster.cells:
                    wx, wy = self.latest_grid.spec.grid_to_world(cell_x, cell_y)
                    frontier_points.points.append(Point(wx, wy, 0.08))
                markers.markers.append(frontier_points)

            text = Marker()
            text.header = marker.header
            text.ns = "voronoi_viewpoint_scores"
            text.id = 1000 + index
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = cluster.subgoal_world[0]
            text.pose.position.y = cluster.subgoal_world[1]
            text.pose.position.z = 0.45
            text.pose.orientation.w = 1.0
            text.scale.z = 0.22
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            terms = cluster.score_terms
            text.text = (
                f"V{index:02d} S={cluster.score:.2f} F={len(cluster.cells)}\n"
                f"I={terms.get('information', 0.0):.2f} "
                f"D={terms.get('distance', 0.0):.2f} "
                f"P={terms.get('previous_subgoal', 0.0):.2f} "
                f"X={terms.get('failure_penalty', 0.0):.2f}"
            )
            markers.markers.append(text)
        if self.state.active_goal is not None:
            goal = self.state.active_goal
            arrow = Marker()
            arrow.header.frame_id = self.map_frame
            arrow.header.stamp = now
            arrow.ns = "active_subgoal_arrow"
            arrow.id = 9000
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = goal.point[0]
            arrow.pose.position.y = goal.point[1]
            arrow.pose.position.z = 0.16
            qz, qw = self._quaternion_z_w_from_yaw(goal.yaw)
            arrow.pose.orientation.z = qz
            arrow.pose.orientation.w = qw
            arrow.scale.x = 0.65
            arrow.scale.y = 0.14
            arrow.scale.z = 0.14
            arrow.color.r = 1.0
            arrow.color.g = 0.12
            arrow.color.b = 0.02
            arrow.color.a = 0.95
            markers.markers.append(arrow)

            label = Marker()
            label.header = arrow.header
            label.ns = "active_subgoal_label"
            label.id = 9001
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = goal.point[0]
            label.pose.position.y = goal.point[1]
            label.pose.position.z = 0.72
            label.pose.orientation.w = 1.0
            label.scale.z = 0.24
            label.color.r = 1.0
            label.color.g = 0.2
            label.color.b = 0.05
            label.color.a = 1.0
            label.text = f"ACTIVE yaw={goal.yaw:.2f}"
            markers.markers.append(label)
        self.frontier_pub.publish(markers)

    def _should_run_initial_spin(self) -> bool:
        if self.initial_spin_done:
            return False
        if self.state.active_goal is not None:
            self.initial_spin_done = True
            self.initial_spin_reason = "skipped_active_goal"
            return False
        if self.robot_yaw is None:
            return True
        if self.initial_spin_done_time > 0.0:
            return time.time() - self.initial_spin_done_time < self.initial_spin_settle_sec
        return True

    def _tick_initial_spin(self) -> None:
        now = time.time()
        if self.robot_yaw is None:
            self.initial_spin_reason = "waiting_for_yaw"
            return
        if not self.initial_spin_active:
            self.initial_spin_active = True
            self.initial_spin_start_time = now
            self.initial_spin_last_yaw = self.robot_yaw
            self.initial_spin_accumulated_yaw = 0.0
            self.initial_spin_reason = "spinning"
            rospy.loginfo("[explore_py] initial 360deg scan started")
        else:
            delta = self._signed_angle_diff(self.robot_yaw, self.initial_spin_last_yaw)
            self.initial_spin_accumulated_yaw += abs(delta)
            self.initial_spin_last_yaw = self.robot_yaw

        timed_out = now - self.initial_spin_start_time >= self.initial_spin_timeout_sec
        completed = self.initial_spin_accumulated_yaw >= self.initial_spin_angle_rad
        if completed or timed_out:
            self._publish_zero_cmd_vel()
            self.initial_spin_done = True
            self.initial_spin_active = False
            self.initial_spin_done_time = now
            self.initial_spin_reason = "completed" if completed else "timeout"
            rospy.loginfo(
                "[explore_py] initial scan finished reason=%s accumulated_yaw=%.2f",
                self.initial_spin_reason,
                self.initial_spin_accumulated_yaw,
            )
            return

        self._publish_initial_spin_cmd()

    def _initial_spin_cmd_timer_callback(self, _event) -> None:
        if self.initial_spin_active and not self.initial_spin_done and not rospy.is_shutdown():
            self._publish_initial_spin_cmd()

    def _publish_initial_spin_cmd(self) -> None:
        cmd = Twist()
        cmd.angular.z = self.initial_spin_angular_speed
        self.cmd_vel_pub.publish(cmd)

    def _publish_zero_cmd_vel(self) -> None:
        self.cmd_vel_pub.publish(Twist())

    def _cancel_move_base_goal(self, reason: str) -> None:
        self.cancel_pub.publish(GoalID())
        self._publish_zero_cmd_vel()
        self.active_move_base_goal_id = ""
        rospy.loginfo("[explore_py] canceled move_base goal after explorer transition: %s", reason)

    def _status_belongs_to_active_goal(self, goal_id: str, goal_stamp: float) -> bool:
        if self.state.active_goal is None:
            return False
        if self.active_move_base_goal_id:
            return goal_id == self.active_move_base_goal_id
        if not goal_id:
            return False
        if self.active_goal_publish_ros_time <= 0.0 or goal_stamp <= 0.0:
            return True
        return goal_stamp >= self.active_goal_publish_ros_time - 0.25

    @staticmethod
    def _goal_id_stamp(status) -> float:
        try:
            stamp = getattr(status.goal_id, "stamp", None)
            return float(stamp.to_sec()) if stamp is not None else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _signed_angle_diff(a: float, b: float) -> float:
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def _active_goal_distance(self):
        if self.state.active_goal is None or self.robot_xy is None:
            return float("inf")
        point = self.state.active_goal.point
        return math.hypot(point[0] - self.robot_xy[0], point[1] - self.robot_xy[1])

    def _convert_grid(self, msg):
        spec = GridSpec(
            width=int(msg.info.width),
            height=int(msg.info.height),
            resolution=float(msg.info.resolution),
            origin_x=float(msg.info.origin.position.x),
            origin_y=float(msg.info.origin.position.y),
            frame_id=msg.header.frame_id or self.map_frame,
        )
        if msg.header.frame_id:
            self.map_frame = msg.header.frame_id
        return OccupancyGridData(spec=spec, data=[int(value) for value in msg.data])

    @staticmethod
    def _cluster_to_dict(cluster, grid=None, include_cells=False):
        if cluster is None:
            return None
        payload = {
            "cluster_id": cluster.cluster_id,
            "subgoal_world": list(cluster.subgoal_world),
            "subgoal_yaw": cluster.subgoal_yaw,
            "centroid_world": list(cluster.centroid_world),
            "information_gain": cluster.information_gain,
            "distance_to_robot": cluster.distance_to_robot,
            "unknown_component_area_m2": float(
                getattr(cluster, "unknown_component_area_m2", 0.0) or 0.0
            ),
            "frontier_length_m": float(
                getattr(cluster, "frontier_length_m", 0.0) or 0.0
            ),
            "expected_visible_unknown_area_m2": float(
                getattr(cluster, "expected_visible_unknown_area_m2", 0.0) or 0.0
            ),
            "score": cluster.score,
            "score_terms": cluster.score_terms,
            "cell_count": len(cluster.cells),
        }
        if include_cells and grid is not None:
            payload["frontier_cells_world"] = [
                list(grid.spec.grid_to_world(cell_x, cell_y))
                for cell_x, cell_y in cluster.cells
            ]
        return payload

    @staticmethod
    def _finite_or_none(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _quaternion_z_w_from_yaw(yaw: float) -> tuple[float, float]:
        return math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def main():
    ExplorePyNode()
    rospy.spin()


if __name__ == "__main__":
    main()
