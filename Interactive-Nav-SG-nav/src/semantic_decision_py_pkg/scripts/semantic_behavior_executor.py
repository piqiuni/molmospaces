#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import threading
import time

from semantic_decision_py_pkg.behavior_execution import (
    BehaviorExecutionStateMachine,
    ExecutionConfig,
    STATE_IDLE,
    STATE_VERIFYING,
)
from semantic_decision_py_pkg.ros_compat import patch_roslogging_findcaller_for_py311

patch_roslogging_findcaller_for_py311()

import actionlib
import rospy
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
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
                interaction_timeout_s=float(config.get("interaction_timeout_s", 30.0)),
                verification_timeout_s=float(config.get("verification_timeout_s", 30.0)),
            )
        )
        self.map_frame = str(config.get("map_frame", "tf_frame_map"))
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
        self.move_base = actionlib.SimpleActionClient(
            topics.get("move_base", "/move_base"), MoveBaseAction
        )
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
        with self.lock:
            if not self._matches_active(payload):
                return
            status = str(payload.get("status") or "")
            if status not in {"SUCCEEDED", "FAILED", "CANCELED", "REJECTED"}:
                return
            commands = self.machine.on_explore_result(
                status == "SUCCEEDED", detail=payload
            )
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
        with self.lock:
            reason = self.machine.timeout_reason()
            commands = self.machine.fail_timeout(reason) if reason else []
            state_payload = {
                **self.machine.summary(),
                "decision_id": "" if self.selection is None else self.selection.get("decision_id", ""),
                "timestamp": time.time(),
            }
        self.state_pub.publish(
            String(data=json.dumps(state_payload, ensure_ascii=False, separators=(",", ":")))
        )
        self._dispatch(commands)

    def _dispatch(self, commands: list[dict]) -> None:
        for command in commands:
            kind = command.get("kind")
            if kind == "explore_frontier":
                self._publish_explore_command(command["candidate"])
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

    def _publish_explore_command(self, candidate: dict) -> None:
        payload = {
            "command_id": self._command_id(candidate),
            "decision_id": candidate.get("decision_id", ""),
            "candidate_id": candidate.get("candidate_id", ""),
            "action": "execute_frontier",
            "cluster_id": (candidate.get("metadata") or {}).get(
                "cluster_id", candidate.get("target_id", "")
            ),
        }
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
        }
        self.interaction_command_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )

    def _run_navigation(self, decision_id: str, candidate: dict) -> None:
        ready = self.move_base.wait_for_server(rospy.Duration(30.0))
        if not ready:
            self._handle_navigation_result(decision_id, False, {"reason": "move_base_unavailable"})
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
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.z = math.sin(0.5 * yaw)
        goal.target_pose.pose.orientation.w = math.cos(0.5 * yaw)
        self.move_base.send_goal(goal)
        deadline = time.monotonic() + self.machine.config.navigation_timeout_s
        state = int(self.move_base.get_state())
        while not rospy.is_shutdown() and state not in TERMINAL_STATES and time.monotonic() < deadline:
            time.sleep(0.10)
            state = int(self.move_base.get_state())
        if state not in TERMINAL_STATES:
            self.move_base.cancel_goal()
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

    def _handle_navigation_result(
        self, decision_id: str, success: bool, detail: dict
    ) -> None:
        with self.lock:
            if self.selection is None or str(self.selection.get("decision_id") or "") != decision_id:
                return
            commands = self.machine.on_navigation_result(success, detail=detail)
        self._dispatch(commands)

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
