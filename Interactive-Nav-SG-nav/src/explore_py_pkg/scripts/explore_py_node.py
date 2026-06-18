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
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from explore_py_pkg.frontier_core import FrontierConfig, FrontierExplorerCore, GridSpec, OccupancyGridData
from explore_py_pkg.nav_client import TERMINAL_FAILURE, TERMINAL_SUCCESS, latest_status, status_name
from explore_py_pkg.skill_api import ExplorationSkillApi
from explore_py_pkg.state import ExplorerState, ExplorerStateConfig, SUBGOAL_WAITING
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
        self.last_goal_publish_time = 0.0
        self.last_status_key = ""

        core_config = FrontierConfig(
            free_max=int(frontier_cfg.get("free_max", 20)),
            occupied_min=int(frontier_cfg.get("occupied_min", 50)),
            min_cluster_cells=int(frontier_cfg.get("min_cluster_cells", 3)),
            connect_8=bool(frontier_cfg.get("connect_8", True)),
            candidate_top_k=int(frontier_cfg.get("candidate_top_k", 12)),
            sensor_range_m=float(frontier_cfg.get("sensor_range_m", 5.0)),
            subgoal_search_radius_cells=int(frontier_cfg.get("subgoal_search_radius_cells", 8)),
            information_weight=float(scoring_cfg.get("information_weight", 1.0)),
            distance_weight=float(scoring_cfg.get("distance_weight", 0.55)),
            semantic_weight=float(scoring_cfg.get("semantic_weight", 0.35)),
            llm_weight=float(scoring_cfg.get("llm_weight", 0.8)),
            revisit_penalty=float(scoring_cfg.get("revisit_penalty", 0.6)),
            failure_penalty=float(scoring_cfg.get("failure_penalty", 1.0)),
        )
        state_config = ExplorerStateConfig(
            goal_reach_tolerance_m=float(exploration_cfg.get("goal_reach_tolerance_m", 0.75)),
            goal_timeout_sec=float(exploration_cfg.get("goal_timeout_sec", 45.0)),
            stall_timeout_sec=float(exploration_cfg.get("stall_timeout_sec", 12.0)),
            stall_distance_m=float(exploration_cfg.get("stall_distance_m", 0.15)),
            blacklist_duration_sec=float(exploration_cfg.get("blacklist_duration_sec", 25.0)),
            failed_cluster_retry_sec=float(exploration_cfg.get("failed_cluster_retry_sec", 120.0)),
            frontier_match_distance_m=float(exploration_cfg.get("frontier_match_distance_m", 1.0)),
        )

        self.core = FrontierExplorerCore(core_config)
        self.state = ExplorerState(state_config)
        self.value_fusion = ValueMapFusion()
        self.skill_api = ExplorationSkillApi(self)

        self.latest_grid_msg = None
        self.latest_grid = None
        self.robot_xy = None
        self.latest_clusters = []
        self.last_selected_cluster = None

        self.goal_pub = rospy.Publisher(self.topics.get("goal", "/move_base_simple/goal"), PoseStamped, queue_size=1)
        self.status_pub = rospy.Publisher(self.topics.get("status", "/explore_py/status"), String, queue_size=1)
        self.frontier_pub = rospy.Publisher(self.topics.get("frontiers", "/explore_py/frontiers"), MarkerArray, queue_size=1)
        self.subgoal_pub = rospy.Publisher(
            self.topics.get("current_subgoal", "/explore_py/current_subgoal"), PointStamped, queue_size=1
        )

        rospy.Subscriber(self.topics.get("occupancy_grid", "/struct_mapping/occ_map"), OccupancyGrid, self.occupancy_callback, queue_size=1)
        rospy.Subscriber(self.topics.get("odom", "/odom"), Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(self.topics.get("move_base_status", "/move_base/status"), GoalStatusArray, self.move_base_status_callback, queue_size=10)
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
        rospy.loginfo("[explore_py] initialized: occ=%s goal=%s status=%s",
                      self.topics.get("occupancy_grid", "/struct_mapping/occ_map"),
                      self.topics.get("goal", "/move_base_simple/goal"),
                      self.topics.get("move_base_status", "/move_base/status"))

    def occupancy_callback(self, msg):
        self.latest_grid_msg = msg
        self.latest_grid = self._convert_grid(msg)

    def odom_callback(self, msg):
        self.robot_xy = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))

    def strategy_bias_callback(self, msg):
        self.value_fusion.set_strategy_bias_json(msg.data)

    def object_map_callback(self, msg):
        self.value_fusion.set_object_map_json(msg.data)

    def unified_graph_callback(self, msg):
        self.value_fusion.set_unified_graph_json(msg.data)

    def navigation_hints_callback(self, msg):
        self.value_fusion.set_navigation_hints_json(msg.data)

    def move_base_status_callback(self, msg):
        status = latest_status(msg)
        if status is None or self.state.active_goal is None:
            return
        status_key = f"{getattr(status.goal_id, 'id', '')}:{int(status.status)}:{getattr(msg.header, 'seq', 0)}"
        if status_key == self.last_status_key:
            return
        self.last_status_key = status_key
        code = int(status.status)
        if code in TERMINAL_FAILURE:
            rospy.logwarn("[explore_py] move_base reported %s, replanning next tick", status_name(code))
            self.state.mark_active_failed(f"move_base_{status_name(code).lower()}")
        elif code in TERMINAL_SUCCESS and self._active_goal_distance() <= self.state.config.goal_reach_tolerance_m * 1.5:
            self.state.mark_active_reached()

    def tick(self, _event):
        if self.latest_grid is None or self.robot_xy is None:
            self._publish_status()
            return

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
                self.state.mark_active_covered_if_frontier_gone(has_frontier)
                if self.state.active_goal is not None:
                    self.state.update_goal_progress(self.robot_xy)

        if self.state.active_goal is None:
            cluster = self.compute_next_subgoal(force=True)
            if cluster is not None:
                self._send_goal(cluster)
        else:
            now = time.time()
            if now - self.last_goal_publish_time >= self.goal_republish_interval_sec:
                self._publish_active_goal()

        self._publish_frontiers()
        self._publish_status()

    def compute_next_subgoal(self, force=False):
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
            self.last_selected_cluster = None
            return None
        cluster = self.core.select_next_cluster(
            self.latest_grid,
            self.robot_xy,
            value_provider=self.value_fusion,
            state=self.state,
        )
        self.last_selected_cluster = cluster
        return cluster

    def build_status_payload(self):
        payload = {
            "ready": self.latest_grid is not None and self.robot_xy is not None,
            "robot_xy": list(self.robot_xy) if self.robot_xy is not None else None,
            "frontier_count": len(self.latest_clusters),
            "selected_cluster": self._cluster_to_dict(self.last_selected_cluster),
            "state": self.state.summary(),
            "semantic_inputs": {
                "object_count": len(self.value_fusion.object_map),
                "navigation_hint_count": len(self.value_fusion.navigation_hints),
                "has_llm_value_grid": self.value_fusion.llm_value_grid is not None,
                "strategy_bias": self.value_fusion.strategy_bias,
            },
        }
        if self.state.active_goal is not None:
            payload["active_goal_distance"] = self._active_goal_distance()
        return payload

    def _send_goal(self, cluster):
        self.state.start_goal(cluster, self.robot_xy, goal_id=cluster.cluster_id)
        self._publish_active_goal()
        rospy.loginfo("[explore_py] sent subgoal cluster=%s point=(%.2f, %.2f) score=%.3f reason=%s",
                      cluster.cluster_id,
                      cluster.subgoal_world[0],
                      cluster.subgoal_world[1],
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
        msg.pose.orientation.w = 1.0
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
        self.frontier_pub.publish(markers)

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
            "centroid_world": list(cluster.centroid_world),
            "information_gain": cluster.information_gain,
            "distance_to_robot": cluster.distance_to_robot,
            "score": cluster.score,
            "score_terms": cluster.score_terms,
            "cell_count": len(cluster.cells),
        }


def main():
    ExplorePyNode()
    rospy.spin()


if __name__ == "__main__":
    main()
