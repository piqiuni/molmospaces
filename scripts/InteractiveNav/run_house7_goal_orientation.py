#!/usr/bin/env python3
"""Validate terminal move_base orientation on an all-open House 7 route."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

from run_house7_force_route import (
    DEFAULT_ROUTE_CONFIG,
    load_route,
    patch_roslogging_findcaller_for_py311,
)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def angle_error(first: float, second: float) -> float:
    return abs(normalize_angle(float(first) - float(second)))


def build_orientation_goal(route: dict[str, Any], yaw_offset_rad: float) -> list[float]:
    goal = [float(value) for value in route["far_goal_xyyaw"]]
    goal[2] = normalize_angle(goal[2] + float(yaw_offset_rad))
    return goal


def yaw_from_quaternion(quaternion) -> float:
    siny_cosp = 2.0 * (
        float(quaternion.w) * float(quaternion.z)
        + float(quaternion.x) * float(quaternion.y)
    )
    cosy_cosp = 1.0 - 2.0 * (
        float(quaternion.y) * float(quaternion.y)
        + float(quaternion.z) * float(quaternion.z)
    )
    return math.atan2(siny_cosp, cosy_cosp)


class GoalOrientationRunner:
    def __init__(
        self,
        output_path: Path,
        ready_timeout_s: float,
        navigation_timeout_s: float,
        position_tolerance_m: float,
        yaw_tolerance_rad: float,
        plan_yaw_tolerance_rad: float,
        map_frame_id: str,
    ) -> None:
        import actionlib
        import rospy
        from move_base_msgs.msg import MoveBaseAction
        from nav_msgs.msg import Odometry, Path as NavPath
        from std_msgs.msg import String

        self.rospy = rospy
        self.String = String
        self.output_path = Path(output_path)
        self.ready_timeout_s = float(ready_timeout_s)
        self.navigation_timeout_s = float(navigation_timeout_s)
        self.position_tolerance_m = float(position_tolerance_m)
        self.yaw_tolerance_rad = float(yaw_tolerance_rad)
        self.plan_yaw_tolerance_rad = float(plan_yaw_tolerance_rad)
        self.map_frame_id = str(map_frame_id)
        self.events: list[dict[str, Any]] = []
        self.result: dict[str, Any] = {}
        self._condition = threading.Condition()
        self._latest_odom: tuple[float, float, float, float] | None = None
        self._latest_plan: list[tuple[float, float, float]] = []
        self._phase_pub = rospy.Publisher(
            "/semantic_decision/route_phase", String, queue_size=4, latch=True
        )
        self._odom_sub = rospy.Subscriber("/odom", Odometry, self._odom_callback, queue_size=8)
        self._plan_sub = rospy.Subscriber(
            "/move_base/GlobalPlanner/plan", NavPath, self._plan_callback, queue_size=8
        )
        self._move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)

    def _odom_callback(self, message) -> None:
        pose = message.pose.pose
        stamp = float(message.header.stamp.to_sec())
        with self._condition:
            self._latest_odom = (
                float(pose.position.x),
                float(pose.position.y),
                yaw_from_quaternion(pose.orientation),
                stamp,
            )
            self._condition.notify_all()

    def _plan_callback(self, message) -> None:
        poses = [
            (
                float(pose.pose.position.x),
                float(pose.pose.position.y),
                yaw_from_quaternion(pose.pose.orientation),
            )
            for pose in message.poses
        ]
        with self._condition:
            self._latest_plan = poses
            self._condition.notify_all()

    def _record(self, event: str, **payload: Any) -> None:
        entry = {
            "event_index": len(self.events) + 1,
            "event": event,
            "wall_time": time.time(),
            **payload,
        }
        self.events.append(entry)
        self._phase_pub.publish(
            self.String(data=json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
        )
        self._write()

    def _write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(
                {"result": self.result, "events": self.events},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.ready_timeout_s
        while not self._move_base.wait_for_server(self.rospy.Duration(0.25)):
            if time.monotonic() >= deadline:
                raise TimeoutError("move_base action server did not become ready")
        from nav_msgs.msg import OccupancyGrid

        try:
            self.rospy.wait_for_message(
                "/struct_mapping/occ_map", OccupancyGrid, timeout=self.ready_timeout_s
            )
        except self.rospy.ROSException as exc:
            raise TimeoutError("/struct_mapping/occ_map did not become ready") from exc
        with self._condition:
            while self._latest_odom is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("/odom did not become ready")
                self._condition.wait(timeout=min(remaining, 0.25))

    def run(self, route: dict[str, Any], yaw_offset_rad: float) -> dict[str, Any]:
        from actionlib_msgs.msg import GoalStatus
        from move_base_msgs.msg import MoveBaseGoal

        route_id = str(route["route_id"])
        goal_xyyaw = build_orientation_goal(route, yaw_offset_rad)
        nominal_goal_yaw = float(route["far_goal_xyyaw"][2])
        required_turn_rad = angle_error(goal_xyyaw[2], nominal_goal_yaw)
        started = time.monotonic()
        self.result = {
            "route_id": route_id,
            "status": "RUNNING",
            "goal_xyyaw": goal_xyyaw,
        }
        self._record(
            "orientation_test_started",
            route_id=route_id,
            nominal_goal_yaw=nominal_goal_yaw,
            yaw_offset_rad=float(yaw_offset_rad),
            required_turn_rad=required_turn_rad,
        )
        try:
            if required_turn_rad < 1.0:
                raise ValueError(
                    f"Orientation test requires at least 1 rad of terminal turn, got {required_turn_rad}"
                )
            self.wait_until_ready()
            self._record("system_ready")

            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = self.map_frame_id
            goal.target_pose.header.stamp = self.rospy.Time.now()
            goal.target_pose.pose.position.x = goal_xyyaw[0]
            goal.target_pose.pose.position.y = goal_xyyaw[1]
            goal.target_pose.pose.orientation.z = math.sin(goal_xyyaw[2] * 0.5)
            goal.target_pose.pose.orientation.w = math.cos(goal_xyyaw[2] * 0.5)
            self._record("navigate_started", segment="orientation", goal=goal_xyyaw)
            self._move_base.send_goal(goal)

            terminal_states = {
                GoalStatus.PREEMPTED,
                GoalStatus.SUCCEEDED,
                GoalStatus.ABORTED,
                GoalStatus.REJECTED,
                GoalStatus.RECALLED,
                GoalStatus.LOST,
            }
            deadline = time.monotonic() + self.navigation_timeout_s
            state = int(self._move_base.get_state())
            while state not in terminal_states and time.monotonic() < deadline:
                time.sleep(0.10)
                state = int(self._move_base.get_state())
            if state not in terminal_states:
                self._move_base.cancel_goal()
                raise TimeoutError(
                    f"move_base did not finish within {self.navigation_timeout_s:.1f}s"
                )

            with self._condition:
                final_odom = self._latest_odom
                final_plan = list(self._latest_plan)
            if final_odom is None:
                raise RuntimeError("No final odometry was received")
            final_x, final_y, final_yaw, final_stamp = final_odom
            position_error_m = math.hypot(final_x - goal_xyyaw[0], final_y - goal_xyyaw[1])
            yaw_error_rad = angle_error(final_yaw, goal_xyyaw[2])
            plan_terminal_yaw = final_plan[-1][2] if final_plan else None
            plan_terminal_yaw_error_rad = (
                angle_error(plan_terminal_yaw, goal_xyyaw[2])
                if plan_terminal_yaw is not None
                else None
            )
            action_succeeded = state == GoalStatus.SUCCEEDED
            verified = (
                action_succeeded
                and position_error_m <= self.position_tolerance_m
                and yaw_error_rad <= self.yaw_tolerance_rad
                and plan_terminal_yaw_error_rad is not None
                and plan_terminal_yaw_error_rad <= self.plan_yaw_tolerance_rad
            )
            navigation = {
                "success": action_succeeded,
                "status_code": state,
                "status": self._move_base.get_goal_status_text() or str(state),
                "goal_xyyaw": goal_xyyaw,
                "final_odom_xyyaw": [final_x, final_y, final_yaw],
                "final_odom_stamp": final_stamp,
                "position_error_m": position_error_m,
                "yaw_error_rad": yaw_error_rad,
                "plan_pose_count": len(final_plan),
                "plan_terminal_yaw": plan_terminal_yaw,
                "plan_terminal_yaw_error_rad": plan_terminal_yaw_error_rad,
            }
            self._record("navigate_succeeded" if action_succeeded else "navigate_failed", result=navigation)
            if not verified:
                raise RuntimeError(f"Terminal orientation verification failed: {navigation}")

            self.result = {
                "route_id": route_id,
                "status": "SUCCEEDED",
                "success": True,
                "duration_s": time.monotonic() - started,
                "initial_door_state": "open",
                "nominal_goal_yaw": nominal_goal_yaw,
                "yaw_offset_rad": float(yaw_offset_rad),
                "required_turn_rad": required_turn_rad,
                "navigation": navigation,
                "tolerances": {
                    "position_m": self.position_tolerance_m,
                    "yaw_rad": self.yaw_tolerance_rad,
                    "plan_yaw_rad": self.plan_yaw_tolerance_rad,
                },
            }
            self._record("orientation_test_succeeded", result=self.result)
        except Exception as exc:
            self.result = {
                "route_id": route_id,
                "status": "FAILED",
                "success": False,
                "duration_s": time.monotonic() - started,
                "initial_door_state": "open",
                "goal_xyyaw": goal_xyyaw,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self._record("orientation_test_failed", result=self.result)
        self._write()
        return self.result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-config", type=Path, default=DEFAULT_ROUTE_CONFIG)
    parser.add_argument("--route-id", default="house7_force_route_01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal-yaw-offset-rad", type=float, default=math.pi)
    parser.add_argument("--ready-timeout-s", type=float, default=60.0)
    parser.add_argument("--navigation-timeout-s", type=float, default=180.0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.30)
    parser.add_argument("--yaw-tolerance-rad", type=float, default=0.25)
    parser.add_argument("--plan-yaw-tolerance-rad", type=float, default=0.05)
    parser.add_argument("--map-frame-id", default="tf_frame_map")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_roslogging_findcaller_for_py311()
    import rospy

    rospy.init_node("house7_goal_orientation_test", anonymous=False)
    route = load_route(args.route_config, args.route_id)
    runner = GoalOrientationRunner(
        output_path=args.output,
        ready_timeout_s=args.ready_timeout_s,
        navigation_timeout_s=args.navigation_timeout_s,
        position_tolerance_m=args.position_tolerance_m,
        yaw_tolerance_rad=args.yaw_tolerance_rad,
        plan_yaw_tolerance_rad=args.plan_yaw_tolerance_rad,
        map_frame_id=args.map_frame_id,
    )
    result = runner.run(route, args.goal_yaw_offset_rad)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
