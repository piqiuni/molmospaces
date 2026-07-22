#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import threading
import time

from semantic_decision_py_pkg.behavior_execution import (
    BehaviorExecutionStateMachine,
    ExecutionConfig,
    STATE_APPROACH_INTERACTION,
    STATE_IDLE,
    STATE_NAVIGATING,
    STATE_VERIFYING,
    committed_turn_sign,
    normalize_angle,
    path_lookahead_point,
)
from semantic_decision_py_pkg.behavior_candidates import interaction_group_reached
from semantic_decision_py_pkg.ros_compat import patch_roslogging_findcaller_for_py311

patch_roslogging_findcaller_for_py311()

import actionlib
import rospy
import tf
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.srv import GetPlan
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
        rospy.init_node("semantic_behavior_executor")
        topics = rospy.get_param("~topics", {}) or {}
        config = rospy.get_param("~executor", {}) or {}
        self.machine = BehaviorExecutionStateMachine(
            ExecutionConfig(
                navigation_timeout_s=float(config.get("navigation_timeout_s", 180.0)),
                interaction_navigation_timeout_s=float(
                    config.get("interaction_navigation_timeout_s", 60.0)
                ),
                interaction_timeout_s=float(config.get("interaction_timeout_s", 30.0)),
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
            config.get("rear_goal_exit_angle_rad", 0.65)
        )
        self.rear_goal_rotate_speed_rad_s = float(
            config.get("rear_goal_rotate_speed_rad_s", 0.35)
        )
        self.rear_goal_prerotate_timeout_s = float(
            config.get("rear_goal_prerotate_timeout_s", 12.0)
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
        self.interaction_final_align_enabled = bool(
            config.get("interaction_final_align_enabled", True)
        )
        self.interaction_final_align_max_distance_m = float(
            config.get("interaction_final_align_max_distance_m", 0.12)
        )
        self.interaction_final_align_yaw_tolerance_rad = float(
            config.get("interaction_final_align_yaw_tolerance_rad", 0.15)
        )
        self.interaction_final_align_rotate_speed_rad_s = float(
            config.get("interaction_final_align_rotate_speed_rad_s", 0.30)
        )
        self.interaction_final_align_trigger_delay_s = float(
            config.get("interaction_final_align_trigger_delay_s", 2.0)
        )
        self.interaction_final_align_timeout_s = float(
            config.get("interaction_final_align_timeout_s", 15.0)
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
        self.lock = threading.RLock()
        self.selection: dict | None = None
        self.latest_graph: dict = {}
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
            commands = self.machine.start(selection)
            self._publish_feedback(selection, "STARTED", None, {})
        self._dispatch(commands)

    def _explore_feedback_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        commands = []
        with self.lock:
            if not self._matches_active(payload):
                return
            status = str(payload.get("status") or "")
            if status == "READY":
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
            commands = self.machine.on_interaction_result(
                bool(payload.get("success")), detail=payload
            )
            if self.machine.state == STATE_VERIFYING:
                commands.extend(self._verify_graph_locked())
        self._dispatch(commands)

    def _graph_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            self.latest_graph = payload
            commands = self._verify_graph_locked()
        self._dispatch(commands)

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
                return self.machine.on_target_visibility(
                    target_visible,
                    detail={
                        "node_id": node.get("id"),
                        "target_visible": target_visible,
                        "visible_pixels": visible_pixels,
                        "min_visible_pixels": min_visible_pixels,
                        "graph_revision": self.latest_graph.get(
                            "graph_revision", 0
                        ),
                    },
                )
            interaction = node.get("interaction") or {}
            interaction_command = self.selection.get("interaction_command") or {}
            target_joint_names = list(interaction_command.get("joint_names") or [])
            if target_joint_names:
                group_reached = interaction_group_reached(
                    list(attributes.get("joint_infos") or []),
                    target_joint_names,
                    list(interaction_command.get("close_other_joint_names") or []),
                    float(
                        interaction_command.get("open_fraction_threshold", 0.67)
                        or 0.67
                    ),
                )
                if group_reached is None:
                    return []
                return self.machine.on_graph_state(
                    "open" if group_reached else "unknown",
                    detail={
                        "node_id": node.get("id"),
                        "state": interaction.get("state"),
                        "interaction_group_id": interaction_command.get(
                            "interaction_group_id", "all_joints"
                        ),
                        "joint_names": target_joint_names,
                        "group_reached": group_reached,
                        "graph_revision": self.latest_graph.get(
                            "graph_revision", 0
                        ),
                    },
                )
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
        with self.lock:
            reason = self.machine.timeout_reason()
            cancel_navigation = bool(reason) and self.machine.state in {
                STATE_NAVIGATING,
                STATE_APPROACH_INTERACTION,
            }
            commands = self.machine.fail_timeout(reason) if reason else []
            state_payload = {
                **self.machine.summary(),
                "decision_id": "" if self.selection is None else self.selection.get("decision_id", ""),
                "timestamp": time.time(),
            }
        self.state_pub.publish(
            String(data=json.dumps(state_payload, ensure_ascii=False, separators=(",", ":")))
        )
        if cancel_navigation:
            self.move_base.cancel_goal()
        self._dispatch(commands)

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
        payload = {
            "command_id": self._command_id(candidate),
            "decision_id": candidate.get("decision_id", ""),
            "candidate_id": candidate.get("candidate_id", ""),
            "action": action,
            "cluster_id": (candidate.get("metadata") or {}).get(
                "cluster_id", candidate.get("target_id", "")
            ),
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
        payload = {
            "command_id": self._command_id(candidate),
            "decision_id": candidate.get("decision_id", ""),
            "candidate_id": candidate.get("candidate_id", ""),
            "event_id": f"{candidate.get('decision_id', 'decision')}_interaction",
            "node_id": interaction.get("node_id", candidate.get("target_id", "")),
            "source_object_name": interaction.get(
                "source_object_name", candidate.get("target_name", "")
            ),
            "action": interaction.get("action", "open"),
            "interaction_mode": interaction.get("interaction_mode", "open_close"),
            "interaction_group_id": interaction.get("interaction_group_id", "all"),
            "joint_names": list(interaction.get("joint_names") or []),
            "close_other_joint_names": list(
                interaction.get("close_other_joint_names") or []
            ),
            "close_other_joints": bool(
                interaction.get("close_other_joints", False)
            ),
            "view_profile": interaction.get("view_profile", "default"),
            "view_tilt_rad": float(interaction.get("view_tilt_rad", 0.55) or 0.55),
            "open_fraction_threshold": float(
                interaction.get("open_fraction_threshold", 0.67) or 0.67
            ),
        }
        self.interaction_command_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
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
    ) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        committed_sign = turn_sign
        try:
            while (
                not rospy.is_shutdown()
                and self._navigation_is_current(decision_id)
                and time.monotonic() < deadline
            ):
                pose = self._current_pose(frame_id)
                if pose is None:
                    time.sleep(0.05)
                    continue
                error = normalize_angle(float(target_yaw) - pose[2])
                if abs(error) <= max(0.0, float(tolerance_rad)):
                    return True
                if committed_sign is None:
                    committed_sign = committed_turn_sign(
                        error,
                        self.rear_goal_pi_tie_tolerance_rad,
                        self.rear_goal_pi_turn_sign,
                    )
                self._publish_rotation(
                    float(committed_sign) * abs(float(speed_rad_s))
                )
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
        pose = self._current_pose(frame_id)
        if pose is None:
            return True
        heading_x, heading_y = heading_target_xy or (goal_x, goal_y)
        target_yaw = math.atan2(heading_y - pose[1], heading_x - pose[0])
        error = normalize_angle(target_yaw - pose[2])
        if abs(error) < self.rear_goal_enter_angle_rad:
            return True
        turn_sign = committed_turn_sign(
            error,
            self.rear_goal_pi_tie_tolerance_rad,
            self.rear_goal_pi_turn_sign,
        )
        return self._rotate_to_yaw(
            decision_id,
            frame_id,
            target_yaw,
            self.rear_goal_exit_angle_rad,
            self.rear_goal_rotate_speed_rad_s,
            self.rear_goal_prerotate_timeout_s,
            turn_sign=turn_sign,
        )

    def _final_align_interaction_goal(
        self,
        decision_id: str,
        frame_id: str,
        goal_x: float,
        goal_y: float,
        goal_yaw: float,
    ) -> bool | None:
        if not self.interaction_final_align_enabled:
            return None
        pose = self._current_pose(frame_id)
        if pose is None:
            return None
        distance = math.hypot(goal_x - pose[0], goal_y - pose[1])
        if distance > self.interaction_final_align_max_distance_m:
            return None
        if abs(normalize_angle(goal_yaw - pose[2])) <= self.interaction_final_align_yaw_tolerance_rad:
            return True
        self.move_base.cancel_goal()
        time.sleep(0.15)
        return self._rotate_to_yaw(
            decision_id,
            frame_id,
            goal_yaw,
            self.interaction_final_align_yaw_tolerance_rad,
            self.interaction_final_align_rotate_speed_rad_s,
            self.interaction_final_align_timeout_s,
        )

    def _run_navigation(self, decision_id: str, candidate: dict) -> None:
        ready = self.move_base.wait_for_server(rospy.Duration(30.0))
        if not ready:
            self._handle_navigation_result(decision_id, False, {"reason": "move_base_unavailable"})
            return
        if not self._navigation_is_current(decision_id):
            return
        goal_values = list(candidate.get("goal_xyyaw") or [])
        if len(goal_values) < 2:
            self._handle_navigation_result(decision_id, False, {"reason": "missing_goal"})
            return
        x, y = float(goal_values[0]), float(goal_values[1])
        yaw = float(goal_values[2]) if len(goal_values) > 2 else 0.0
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = str(
            (candidate.get("metadata") or {}).get("frame_id") or self.map_frame
        )
        goal_frame = goal.target_pose.header.frame_id
        plan_reachable, path_lookahead = self._preflight_navigation_plan(
            goal_frame, x, y, yaw
        )
        if not plan_reachable:
            self._handle_navigation_result(
                decision_id,
                False,
                {"reason": "make_plan_unreachable"},
            )
            return
        self._prerotate_for_rear_goal(
            decision_id,
            goal_frame,
            x,
            y,
            heading_target_xy=path_lookahead,
        )
        if not self._navigation_is_current(decision_id):
            return
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.z = math.sin(0.5 * yaw)
        goal.target_pose.pose.orientation.w = math.cos(0.5 * yaw)
        self.move_base.send_goal(goal)
        navigation_timeout_s = (
            self.machine.config.interaction_navigation_timeout_s
            if str(candidate.get("behavior_type") or "") == "INTERACT"
            else self.machine.config.navigation_timeout_s
        )
        deadline = time.monotonic() + navigation_timeout_s
        near_goal_since = None
        interaction_navigation = str(candidate.get("behavior_type") or "") == "INTERACT"
        state = int(self.move_base.get_state())
        while (
            not rospy.is_shutdown()
            and self._navigation_is_current(decision_id)
            and state not in TERMINAL_STATES
            and time.monotonic() < deadline
        ):
            time.sleep(0.10)
            state = int(self.move_base.get_state())
            if interaction_navigation:
                pose = self._current_pose(goal_frame)
                if pose is None:
                    near_goal_since = None
                    continue
                distance = math.hypot(x - pose[0], y - pose[1])
                yaw_error = abs(normalize_angle(yaw - pose[2]))
                if (
                    distance <= self.interaction_final_align_max_distance_m
                    and yaw_error > self.interaction_final_align_yaw_tolerance_rad
                ):
                    if near_goal_since is None:
                        near_goal_since = time.monotonic()
                    elif (
                        time.monotonic() - near_goal_since
                        >= self.interaction_final_align_trigger_delay_s
                    ):
                        aligned = self._final_align_interaction_goal(
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
            if interaction_navigation:
                aligned = self._final_align_interaction_goal(
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
            self._handle_navigation_result(decision_id, False, {"reason": "navigation_timeout"})
            return
        self._handle_navigation_result(
            decision_id,
            state == GoalStatus.SUCCEEDED,
            {
                "status_code": state,
                "status": self.move_base.get_goal_status_text() or str(state),
            },
        )

    def _preflight_navigation_plan(
        self,
        frame_id: str,
        goal_x: float,
        goal_y: float,
        goal_yaw: float,
    ) -> tuple[bool, tuple[float, float] | None]:
        if not self.make_plan_preflight_enabled:
            return True, None
        pose = self._current_pose(frame_id)
        if pose is None:
            return self.make_plan_fail_open, None
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
            return self.make_plan_fail_open, None
        poses = list(response.plan.poses or [])
        if not poses:
            return False, None
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
        return reachable, lookahead

    def _handle_navigation_result(
        self, decision_id: str, success: bool, detail: dict
    ) -> None:
        with self.lock:
            if self.selection is None or str(self.selection.get("decision_id") or "") != decision_id:
                return
            commands = self.machine.on_navigation_result(success, detail=detail)
            if self.machine.state == STATE_VERIFYING:
                commands.extend(self._verify_graph_locked())
        self._dispatch(commands)

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
            status = "SUCCEEDED" if command.get("success") else "FAILED"
            detail = dict(command.get("detail") or {})
            self._publish_feedback(selection, status, bool(command.get("success")), detail)
            self.selection = None
            self.machine.reset()

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
