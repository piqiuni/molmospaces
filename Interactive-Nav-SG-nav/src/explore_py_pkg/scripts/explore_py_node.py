#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import time
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
from std_msgs.msg import String
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
        self.latest_global_plan_pose_count = 0
        self.latest_global_plan_length_m = 0.0
        self.latest_global_plan_time = 0.0
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

        core_config = FrontierConfig(
            free_max=int(frontier_cfg.get("free_max", 20)),
            occupied_min=int(frontier_cfg.get("occupied_min", 50)),
            hard_min_cluster_cells=int(frontier_cfg.get("hard_min_cluster_cells", 3)),
            min_cluster_cells=int(frontier_cfg.get("min_cluster_cells", 3)),
            connect_8=bool(frontier_cfg.get("connect_8", True)),
            candidate_top_k=int(frontier_cfg.get("candidate_top_k", 12)),
            sensor_range_m=float(frontier_cfg.get("sensor_range_m", 5.0)),
            subgoal_search_radius_cells=int(frontier_cfg.get("subgoal_search_radius_cells", 8)),
            min_subgoal_distance_m=float(frontier_cfg.get("min_subgoal_distance_m", 0.75)),
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
            near_frontier_relax_distance_m=float(frontier_cfg.get("near_frontier_relax_distance_m", 1.5)),
            relaxed_min_viewpoint_frontier_distance_m=float(
                frontier_cfg.get("relaxed_min_viewpoint_frontier_distance_m", 0.35)
            ),
        )
        state_config = ExplorerStateConfig(
            goal_reach_tolerance_m=float(exploration_cfg.get("goal_reach_tolerance_m", 0.75)),
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
            failed_point_soft_blacklist_sec=float(exploration_cfg.get("failed_point_soft_blacklist_sec", 45.0)),
            failed_point_blacklist_sec=float(exploration_cfg.get("failed_point_blacklist_sec", 180.0)),
            failed_point_blacklist_radius_m=float(exploration_cfg.get("failed_point_blacklist_radius_m", 1.25)),
            reached_point_blacklist_sec=float(exploration_cfg.get("reached_point_blacklist_sec", 90.0)),
            reached_point_blacklist_radius_m=float(exploration_cfg.get("reached_point_blacklist_radius_m", 0.75)),
        )

        self.core = FrontierExplorerCore(core_config)
        self.state = ExplorerState(state_config)
        self.value_fusion = ValueMapFusion()
        self.skill_api = ExplorationSkillApi(self)

        self.latest_grid_msg = None
        self.latest_grid = None
        self.robot_xy = None
        self.robot_yaw = None
        self.latest_clusters = []
        self.last_selected_cluster = None
        self.preplanned_cluster = None

        self.goal_pub = rospy.Publisher(self.topics.get("goal", "/move_base_simple/goal"), PoseStamped, queue_size=1)
        self.cancel_pub = rospy.Publisher(self.topics.get("move_base_cancel", "/move_base/cancel"), GoalID, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher(self.topics.get("cmd_vel", "/cmd_vel"), Twist, queue_size=1)
        self.status_pub = rospy.Publisher(self.topics.get("status", "/explore_py/status"), String, queue_size=1)
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

        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.tick_rate_hz, 1e-3)), self.tick)
        self.initial_spin_cmd_timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.initial_spin_cmd_rate_hz, 1e-3)),
            self._initial_spin_cmd_timer_callback,
        )
        rospy.loginfo("[explore_py] initialized: occ=%s goal=%s status=%s",
                      self.topics.get("occupancy_grid", "/struct_mapping/occ_map"),
                      self.topics.get("goal", "/move_base_simple/goal"),
                      self.topics.get("move_base_status", "/move_base/status"))

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

    def global_plan_callback(self, msg: NavPath):
        self.latest_global_plan_pose_count = len(msg.poses)
        self.latest_global_plan_length_m = self._path_length(msg)
        self.latest_global_plan_time = time.time()

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
                self.preplanned_cluster = None
                return
            if code in TERMINAL_SUCCESS and self._active_goal_distance() <= self.state.config.goal_reach_tolerance_m * 1.5:
                self.state.mark_active_reached()
                self.active_move_base_goal_id = ""
                self.preplanned_cluster = None
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

        active_goal_before_update = self.state.active_goal is not None
        if self.state.active_goal is not None:
            if not self.core.is_free_world(self.latest_grid, self.state.active_goal.point):
                self.state.fail_active_if_goal_not_free(False)
            else:
                has_frontier = self.core.has_frontier_near(
                    self.latest_grid,
                    self.state.active_goal.point,
                    self.state.config.frontier_match_distance_m,
                    min_cells=self.active_goal_frontier_min_cells,
                )
                progress = self.state.update_goal_progress(self.robot_xy, robot_yaw=self.robot_yaw)
                if progress == SUBGOAL_REACHED:
                    if has_frontier:
                        self.state.mark_active_reached_pose_only()
                    else:
                        self.state.mark_active_reached()
                elif self.state.active_goal is not None:
                    self._fail_if_local_plan_missing()
                if self.state.active_goal is not None:
                    self.state.mark_active_covered_if_frontier_gone(
                        has_frontier,
                        confirm_ticks=self.frontier_gone_confirm_ticks,
                        min_goal_age_sec=self.frontier_gone_min_goal_age_sec,
                    )
        if active_goal_before_update and self.state.active_goal is None:
            self.preplanned_cluster = None
            self._cancel_move_base_goal(self.state.last_event or "explorer_closed_goal")
            self.active_move_base_goal_id = ""

        if self.state.active_goal is None:
            cluster = self.preplanned_cluster or self.compute_next_subgoal(force=True)
            self.preplanned_cluster = None
            if cluster is not None:
                self._send_goal(cluster)
        else:
            self.preplanned_cluster = self.compute_next_subgoal(force=False, publish_selection=False)
            now = time.time()
            if self.goal_republish_interval_sec > 0.0 and now - self.last_goal_publish_time >= self.goal_republish_interval_sec:
                self._publish_active_goal()

        self._publish_frontiers()
        self._publish_status()

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
        if self.local_plan_bad_since <= 0.0:
            self.local_plan_bad_since = now
            return
        bad_duration = now - self.local_plan_bad_since
        if bad_duration >= self.local_plan_watchdog_sec:
            rospy.logwarn(
                "[explore_py] local plan watchdog failed active goal: global_poses=%d local_poses=%d local_len=%.2fm bad_duration=%.1fs",
                self.latest_global_plan_pose_count,
                self.latest_local_plan_pose_count,
                self.latest_local_plan_length_m,
                bad_duration,
            )
            self.state.mark_active_failed("local_plan_degenerate", source="explorer")
            self.local_plan_bad_since = 0.0

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
        cluster = self.core.select_next_cluster(
            self.latest_grid,
            self.robot_xy,
            value_provider=self.value_fusion,
            state=self.state,
        )
        if publish_selection:
            self.last_selected_cluster = cluster
        return cluster

    def build_status_payload(self):
        payload = {
            "ready": self.latest_grid is not None and self.robot_xy is not None,
            "robot_xy": list(self.robot_xy) if self.robot_xy is not None else None,
            "frontier_count": len(self.latest_clusters),
            "frontier_debug": self.core.last_debug_stats,
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
                "local_plan_poses": self.latest_local_plan_pose_count,
                "local_plan_length_m": self.latest_local_plan_length_m,
                "local_plan_bad_since": self.local_plan_bad_since,
            },
        }
        if self.state.active_goal is not None:
            payload["active_goal_distance"] = self._active_goal_distance()
        if self.last_move_base_feedback is not None:
            payload["move_base_feedback"] = self.last_move_base_feedback
        return payload

    def _send_goal(self, cluster):
        self.local_plan_bad_since = 0.0
        self.latest_global_plan_pose_count = 0
        self.latest_global_plan_length_m = 0.0
        self.latest_local_plan_pose_count = 0
        self.latest_local_plan_length_m = 0.0
        self.active_move_base_goal_id = ""
        self.active_goal_publish_ros_time = 0.0
        self.active_goal_publish_wall_time = 0.0
        self.state.start_goal(cluster, self.robot_xy, robot_yaw=self.robot_yaw, goal_id=cluster.cluster_id)
        self._publish_status()
        self._publish_active_goal()
        rospy.loginfo("[explore_py] sent subgoal cluster=%s point=(%.2f, %.2f, yaw=%.2f) score=%.3f reason=%s",
                      cluster.cluster_id,
                      cluster.subgoal_world[0],
                      cluster.subgoal_world[1],
                      cluster.subgoal_yaw,
                      cluster.score,
                      json.dumps(cluster.score_terms, sort_keys=True))

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
        # Keep move_base position-only for exploration. The observation yaw is
        # tracked in explorer state/debug output and can be executed separately.
        msg.pose.orientation.w = 1.0
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
        for index, cluster in enumerate(self.latest_clusters[:100]):
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = now
            marker.ns = "frontier_clusters"
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = cluster.subgoal_world[0]
            marker.pose.position.y = cluster.subgoal_world[1]
            marker.pose.position.z = 0.05
            marker.pose.orientation.w = 1.0
            scale = max(0.12, min(0.8, math.sqrt(len(cluster.cells)) * 0.06))
            marker.scale.x = marker.scale.y = marker.scale.z = scale
            marker.color.r = 0.1
            marker.color.g = max(0.2, min(1.0, cluster.score))
            marker.color.b = 1.0 - marker.color.g * 0.4
            marker.color.a = 0.85
            markers.markers.append(marker)

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
            text.ns = "frontier_cluster_labels"
            text.id = 1000 + index
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = cluster.subgoal_world[0]
            text.pose.position.y = cluster.subgoal_world[1]
            text.pose.position.z = 0.45
            text.pose.orientation.w = 1.0
            text.scale.z = 0.22
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = f"{index}:{cluster.score:.2f}"
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
    def _cluster_to_dict(cluster):
        if cluster is None:
            return None
        return {
            "cluster_id": cluster.cluster_id,
            "subgoal_world": list(cluster.subgoal_world),
            "subgoal_yaw": cluster.subgoal_yaw,
            "centroid_world": list(cluster.centroid_world),
            "information_gain": cluster.information_gain,
            "distance_to_robot": cluster.distance_to_robot,
            "score": cluster.score,
            "score_terms": cluster.score_terms,
            "cell_count": len(cluster.cells),
        }

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
