#!/usr/bin/env python3
"""Execute one frozen House 7 navigation-force-interaction-navigation route."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTE_CONFIG = (
    REPO_ROOT
    / "scripts"
    / "InteractiveNav"
    / "configs"
    / "semantic_decision"
    / "house7_force_routes.yaml"
)


def patch_roslogging_findcaller_for_py311() -> None:
    import sys

    if sys.version_info < (3, 11):
        return
    try:
        import logging
        import rosgraph.roslogging as roslogging
    except Exception:
        return
    if getattr(roslogging.RospyLogger.findCaller, "_house7_route_safe", False):
        return

    def safe_find_caller(self, *args, **kwargs):
        result = logging.Logger.findCaller(self, *args, **kwargs)
        if len(result) == 3:
            return result[0], result[1], result[2], None
        return result

    safe_find_caller._house7_route_safe = True
    roslogging.RospyLogger.findCaller = safe_find_caller


def load_route(path: Path, route_id: str) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    routes = list(payload.get("routes") or [])
    if not routes:
        raise ValueError(f"No routes found in {path}")
    if not route_id:
        return routes[0]
    matches = [route for route in routes if route.get("route_id") == route_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one route_id={route_id!r}, found {len(matches)} in {path}"
        )
    return matches[0]


def portal_snapshot(graph: dict[str, Any] | None, source_object_name: str) -> dict[str, Any] | None:
    if not graph:
        return None
    for node in graph.get("nodes") or []:
        attributes = node.get("attributes") or {}
        if (
            attributes.get("source_object_name") == source_object_name
            or node.get("name") == source_object_name
        ):
            interaction = node.get("interaction") or {}
            return {
                "node_id": node.get("id"),
                "state": interaction.get("state", "unknown"),
                "traversable": interaction.get("traversable"),
                "requires_interaction": interaction.get("requires_interaction"),
                "operation_history": list(interaction.get("operation_history") or []),
                "graph_revision": graph.get("graph_revision"),
                "room_count": sum(
                    1 for candidate in graph.get("nodes") or [] if candidate.get("type") == "room"
                ),
            }
    return None


def portal_state_matches(snapshot: dict[str, Any] | None, expected_state: str) -> bool:
    if snapshot is None:
        return False
    if snapshot.get("state") == expected_state:
        return True
    # With minimal GT, the graph must not infer a closed state from simulator-only
    # flags. Before the executor reports an interaction result, an observed blocked
    # portal is therefore represented as unknown + non-traversable.
    return bool(
        expected_state == "closed"
        and snapshot.get("state") == "unknown"
        and snapshot.get("traversable") is False
        and snapshot.get("requires_interaction") is True
    )


def door_exit_goal(route: dict[str, Any], standoff_m: float = 0.8) -> list[float]:
    explicit = route.get("door_exit_xyyaw")
    if explicit is not None:
        return [float(value) for value in explicit]
    center_x, center_y = (float(value) for value in route["portal_center_xy"])
    normal_x, normal_y = (float(value) for value in route["portal_normal_xy"])
    start_x, start_y = (float(value) for value in route["start_xyyaw"][:2])
    start_side = 1.0 if (start_x - center_x) * normal_x + (start_y - center_y) * normal_y >= 0 else -1.0
    cross_x = center_x - start_side * normal_x * float(standoff_m)
    cross_y = center_y - start_side * normal_y * float(standoff_m)
    cross_yaw = math.atan2(-start_side * normal_y, -start_side * normal_x)
    return [cross_x, cross_y, cross_yaw]


def post_interaction_goals(route: dict[str, Any]) -> list[list[float]]:
    crossing = door_exit_goal(route)
    goals = [crossing]
    path = [[float(value) for value in point[:2]] for point in route.get("post_interaction_path_xy", [])]
    if not path:
        return goals
    final_xy = [float(value) for value in route["far_goal_xyyaw"][:2]]
    center_x, center_y = (float(value) for value in route["portal_center_xy"])
    normal_x, normal_y = (float(value) for value in route["portal_normal_xy"])
    start_side = float(
        (route.get("validation") or {}).get(
            "start_side_sign",
            1.0
            if (float(route["start_xyyaw"][0]) - center_x) * normal_x
            + (float(route["start_xyyaw"][1]) - center_y) * normal_y
            >= 0
            else -1.0,
        )
    )
    previous_xy = crossing[:2]
    intermediate = []
    for index, point in enumerate(path):
        side_projection = ((point[0] - center_x) * normal_x + (point[1] - center_y) * normal_y) * start_side
        if side_projection >= -0.10:
            continue
        if math.dist(point, previous_xy) < 0.35 or math.dist(point, final_xy) < 0.25:
            continue
        next_xy = path[index + 1] if index + 1 < len(path) else final_xy
        intermediate.append([point[0], point[1], math.atan2(next_xy[1] - point[1], next_xy[0] - point[0])])
        previous_xy = point
    return goals + intermediate


def route_visualization_plan(route: dict[str, Any]) -> dict[str, Any]:
    subgoals = [
        {"segment": "approach", "goal_xyyaw": list(route["door_approach_xyyaw"])},
    ]
    for index, goal in enumerate(post_interaction_goals(route)):
        subgoals.append(
            {
                "segment": "crossing" if index == 0 else f"post_interaction_{index:02d}",
                "goal_xyyaw": list(goal),
            }
        )
    subgoals.append({"segment": "final", "goal_xyyaw": list(route["far_goal_xyyaw"])})
    return {
        "subgoals": subgoals,
        "interaction_goal_xyyaw": list(route["door_approach_xyyaw"]),
        "interaction_object_id": str((route.get("interaction") or {}).get("object_id") or ""),
    }


class StagedRouteExecutor:
    def __init__(self, backend, output_path: Path | None = None) -> None:
        self.backend = backend
        self.output_path = Path(output_path) if output_path else None
        self.events: list[dict[str, Any]] = []
        self.result: dict[str, Any] = {}

    def _record(self, event: str, **payload: Any) -> dict[str, Any]:
        entry = {
            "event_index": len(self.events) + 1,
            "event": event,
            "wall_time": time.time(),
            **payload,
        }
        self.events.append(entry)
        self.backend.publish_phase(entry)
        self._write()
        return entry

    def _write(self) -> None:
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(
                {"result": self.result, "events": self.events},
                ensure_ascii=False,
                indent=2,
            )
        )

    def run(self, route: dict[str, Any]) -> dict[str, Any]:
        route_id = str(route["route_id"])
        target_root = str(route["interaction"]["object_id"])
        started = time.monotonic()
        self.result = {"route_id": route_id, "status": "RUNNING"}
        self._record(
            "route_started",
            route_id=route_id,
            target_root=target_root,
            route_plan=route_visualization_plan(route),
        )
        try:
            self.backend.wait_until_ready()
            self._record("system_ready")

            approach_goal = list(route["door_approach_xyyaw"])
            self._record("navigate_started", segment="approach", goal=approach_goal)
            approach_result = self.backend.navigate(approach_goal, segment="approach")
            if not approach_result.get("success"):
                raise RuntimeError(f"Approach navigation failed: {approach_result}")
            self._record("navigate_succeeded", segment="approach", result=approach_result)

            closed_portal = self.backend.wait_for_portal_state(target_root, "closed")
            self._record("closed_portal_verified", portal=closed_portal)

            command = {
                "command_id": f"{route_id}_open",
                "candidate_id": target_root,
                "decision_id": f"{route_id}_fixed_sequence",
                "event_id": f"{route_id}_interaction",
                "object_id": target_root,
                "action": "open",
                "interaction_mode": "open_close",
            }
            self._record("interaction_started", command=command)
            interaction_result = self.backend.interact(command)
            if not interaction_result.get("success"):
                raise RuntimeError(f"Interaction failed: {interaction_result}")
            self._record("interaction_succeeded", result=interaction_result)

            open_portal = self.backend.wait_for_portal_state(target_root, "open")
            self._record("open_portal_verified", portal=open_portal)

            self._record("navigation_map_settle_started")
            map_settle_result = self.backend.wait_for_navigation_map()
            self._record("navigation_map_settled", result=map_settle_result)

            waypoint_results = []
            waypoint_map_settles = []
            for index, waypoint_goal in enumerate(post_interaction_goals(route)):
                segment = "crossing" if index == 0 else f"post_interaction_{index:02d}"
                self._record("navigate_started", segment=segment, goal=waypoint_goal)
                waypoint_result = self.backend.navigate(waypoint_goal, segment=segment)
                if not waypoint_result.get("success"):
                    raise RuntimeError(f"Post-interaction navigation failed: {waypoint_result}")
                self._record("navigate_succeeded", segment=segment, result=waypoint_result)
                waypoint_results.append(waypoint_result)

                self._record("navigation_map_settle_started", segment=segment)
                waypoint_map_settle = self.backend.wait_for_navigation_map()
                self._record(
                    "navigation_map_settled",
                    segment=segment,
                    result=waypoint_map_settle,
                )
                waypoint_map_settles.append(waypoint_map_settle)

            final_goal = list(route["far_goal_xyyaw"])
            self._record("navigate_started", segment="final", goal=final_goal)
            final_result = self.backend.navigate(final_goal, segment="final")
            if not final_result.get("success"):
                raise RuntimeError(f"Final navigation failed: {final_result}")
            self._record("navigate_succeeded", segment="final", result=final_result)

            final_graph = self.backend.latest_graph_summary(target_root)
            duration_s = time.monotonic() - started
            self.result = {
                "route_id": route_id,
                "status": "SUCCEEDED",
                "success": True,
                "duration_s": duration_s,
                "approach_navigation": approach_result,
                "interaction": interaction_result,
                "navigation_map_settle": map_settle_result,
                "post_interaction_navigation": waypoint_results,
                "post_interaction_map_settles": waypoint_map_settles,
                "final_navigation": final_result,
                "closed_portal": closed_portal,
                "open_portal": open_portal,
                "final_graph": final_graph,
            }
            self._record("route_succeeded", result=self.result)
        except Exception as exc:
            self.result = {
                "route_id": route_id,
                "status": "FAILED",
                "success": False,
                "duration_s": time.monotonic() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self._record("route_failed", result=self.result)
        self._write()
        return self.result


class RosRouteBackend:
    def __init__(
        self,
        navigation_timeout_s: float = 180.0,
        interaction_timeout_s: float = 30.0,
        graph_timeout_s: float = 30.0,
        ready_timeout_s: float = 60.0,
        map_settle_updates: int = 4,
        map_settle_timeout_s: float = 30.0,
        map_frame_id: str = "tf_frame_map",
        interaction_command_topic: str = "/semantic_decision/interaction_command",
        interaction_result_topic: str = "/semantic_mapping/interaction_result",
        unified_graph_topic: str = "/semantic_mapping/unified_graph",
        route_phase_topic: str = "/semantic_decision/route_phase",
    ) -> None:
        import actionlib
        import rospy
        from move_base_msgs.msg import MoveBaseAction
        from nav_msgs.msg import OccupancyGrid
        from std_msgs.msg import String

        self.rospy = rospy
        self.String = String
        self.navigation_timeout_s = float(navigation_timeout_s)
        self.interaction_timeout_s = float(interaction_timeout_s)
        self.graph_timeout_s = float(graph_timeout_s)
        self.ready_timeout_s = float(ready_timeout_s)
        self.map_settle_updates = max(1, int(map_settle_updates))
        self.map_settle_timeout_s = float(map_settle_timeout_s)
        self.map_frame_id = str(map_frame_id)
        self._condition = threading.Condition()
        self._latest_graph: dict[str, Any] | None = None
        self._interaction_results: dict[str, dict[str, Any]] = {}
        self._raw_map_stamps: list[float] = []
        self._planning_map_stamps: list[float] = []
        self._phase_pub = rospy.Publisher(route_phase_topic, String, queue_size=4, latch=True)
        self._interaction_pub = rospy.Publisher(
            interaction_command_topic, String, queue_size=4, latch=True
        )
        self._interaction_sub = rospy.Subscriber(
            interaction_result_topic, String, self._interaction_callback, queue_size=8
        )
        self._graph_sub = rospy.Subscriber(
            unified_graph_topic, String, self._graph_callback, queue_size=4
        )
        self._raw_map_sub = rospy.Subscriber(
            "/struct_mapping/occ_map", OccupancyGrid, self._raw_map_callback, queue_size=4
        )
        self._planning_map_sub = rospy.Subscriber(
            "/semantic_mapping/planning_occ_map",
            OccupancyGrid,
            self._planning_map_callback,
            queue_size=4,
        )
        self._move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)

    def _interaction_callback(self, message) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        command_id = str(payload.get("command_id") or "")
        if not command_id:
            return
        with self._condition:
            self._interaction_results[command_id] = payload
            self._condition.notify_all()

    def _graph_callback(self, message) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        with self._condition:
            self._latest_graph = payload
            self._condition.notify_all()

    def _raw_map_callback(self, message) -> None:
        with self._condition:
            self._raw_map_stamps.append(message.header.stamp.to_sec())
            self._condition.notify_all()

    def _planning_map_callback(self, message) -> None:
        with self._condition:
            self._planning_map_stamps.append(message.header.stamp.to_sec())
            self._condition.notify_all()

    def publish_phase(self, payload: dict[str, Any]) -> None:
        self._phase_pub.publish(
            self.String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.ready_timeout_s
        while not self._move_base.wait_for_server(self.rospy.Duration(0.25)):
            if time.monotonic() >= deadline:
                raise TimeoutError("move_base action server did not become ready")
        for topic, message_type in (
            ("/odom", "nav_msgs/Odometry"),
            ("/semantic_mapping/planning_occ_map", "nav_msgs/OccupancyGrid"),
        ):
            package_name, class_name = message_type.split("/")
            module = __import__(f"{package_name}.msg", fromlist=[class_name])
            try:
                self.rospy.wait_for_message(
                    topic,
                    getattr(module, class_name),
                    timeout=self.ready_timeout_s,
                )
            except self.rospy.ROSException as exc:
                raise TimeoutError(f"Required topic did not become ready: {topic}") from exc

    def navigate(self, goal_xyyaw: list[float], segment: str) -> dict[str, Any]:
        from actionlib_msgs.msg import GoalStatus
        from move_base_msgs.msg import MoveBaseGoal

        x, y, yaw = (float(value) for value in goal_xyyaw)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.map_frame_id
        goal.target_pose.header.stamp = self.rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.z = math.sin(yaw * 0.5)
        goal.target_pose.pose.orientation.w = math.cos(yaw * 0.5)
        started = time.monotonic()
        self._move_base.send_goal(goal)
        deadline = started + self.navigation_timeout_s
        terminal_states = {
            GoalStatus.PREEMPTED,
            GoalStatus.SUCCEEDED,
            GoalStatus.ABORTED,
            GoalStatus.REJECTED,
            GoalStatus.RECALLED,
            GoalStatus.LOST,
        }
        state = int(self._move_base.get_state())
        while state not in terminal_states and time.monotonic() < deadline:
            time.sleep(0.10)
            state = int(self._move_base.get_state())
        if state not in terminal_states:
            self._move_base.cancel_goal()
            return {
                "segment": segment,
                "success": False,
                "status": "TIMEOUT",
                "duration_s": time.monotonic() - started,
                "goal": goal_xyyaw,
            }
        return {
            "segment": segment,
            "success": state == GoalStatus.SUCCEEDED,
            "status": self._move_base.get_goal_status_text() or str(state),
            "status_code": state,
            "duration_s": time.monotonic() - started,
            "goal": goal_xyyaw,
        }

    def interact(self, command: dict[str, Any]) -> dict[str, Any]:
        command_id = str(command["command_id"])
        deadline = time.monotonic() + self.interaction_timeout_s
        while self._interaction_pub.get_num_connections() <= 0:
            if time.monotonic() >= deadline:
                raise TimeoutError("No force interaction command subscriber")
            self.rospy.sleep(0.05)
        self._interaction_pub.publish(
            self.String(data=json.dumps(command, ensure_ascii=False, separators=(",", ":")))
        )
        with self._condition:
            while command_id not in self._interaction_results:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"Interaction result timed out: {command_id}")
                self._condition.wait(timeout=min(remaining, 0.5))
            return dict(self._interaction_results[command_id])

    def wait_for_portal_state(
        self, source_object_name: str, expected_state: str
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.graph_timeout_s
        with self._condition:
            while True:
                snapshot = portal_snapshot(self._latest_graph, source_object_name)
                if portal_state_matches(snapshot, expected_state):
                    return snapshot
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"Portal state {expected_state!r} not observed for {source_object_name}; "
                        f"latest={snapshot}"
                    )
                self._condition.wait(timeout=min(remaining, 0.5))

    def latest_graph_summary(self, source_object_name: str) -> dict[str, Any] | None:
        with self._condition:
            return portal_snapshot(self._latest_graph, source_object_name)

    def wait_for_navigation_map(self) -> dict[str, Any]:
        started = time.monotonic()
        deadline = started + self.map_settle_timeout_s
        with self._condition:
            raw_start = len(self._raw_map_stamps)
            planning_start = len(self._planning_map_stamps)
            while True:
                raw_updates = len(self._raw_map_stamps) - raw_start
                planning_updates = len(self._planning_map_stamps) - planning_start
                if raw_updates >= self.map_settle_updates and planning_updates >= 1:
                    return {
                        "raw_map_updates": raw_updates,
                        "raw_map_stamps": self._raw_map_stamps[raw_start:],
                        "planning_map_updates": planning_updates,
                        "planning_map_stamp": self._planning_map_stamps[-1],
                        "duration_s": time.monotonic() - started,
                    }
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "Navigation maps did not settle after interaction; "
                        f"raw_updates={raw_updates}, planning_updates={planning_updates}"
                    )
                self._condition.wait(timeout=min(remaining, 0.5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-config", type=Path, default=DEFAULT_ROUTE_CONFIG)
    parser.add_argument("--route-id", default="house7_force_route_01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--navigation-timeout-s", type=float, default=180.0)
    parser.add_argument("--interaction-timeout-s", type=float, default=30.0)
    parser.add_argument("--graph-timeout-s", type=float, default=30.0)
    parser.add_argument("--ready-timeout-s", type=float, default=60.0)
    parser.add_argument("--map-settle-updates", type=int, default=4)
    parser.add_argument("--map-settle-timeout-s", type=float, default=30.0)
    parser.add_argument("--map-frame-id", default="tf_frame_map")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    startup_path = args.output.with_suffix(args.output.suffix + ".startup.json")

    def mark_startup(stage: str) -> None:
        startup_path.parent.mkdir(parents=True, exist_ok=True)
        startup_path.write_text(
            json.dumps({"stage": stage, "wall_time": time.time()}, indent=2)
        )

    mark_startup("args_parsed")
    print("[house7_route] importing rospy", flush=True)
    patch_roslogging_findcaller_for_py311()
    import rospy

    mark_startup("initializing_node")
    print("[house7_route] initializing ROS node", flush=True)
    rospy.init_node("house7_force_route_runner", anonymous=False)
    mark_startup("node_initialized")
    print("[house7_route] ROS node initialized", flush=True)
    route = load_route(args.route_config, args.route_id)
    mark_startup("route_loaded")
    backend = RosRouteBackend(
        navigation_timeout_s=args.navigation_timeout_s,
        interaction_timeout_s=args.interaction_timeout_s,
        graph_timeout_s=args.graph_timeout_s,
        ready_timeout_s=args.ready_timeout_s,
        map_settle_updates=args.map_settle_updates,
        map_settle_timeout_s=args.map_settle_timeout_s,
        map_frame_id=args.map_frame_id,
    )
    mark_startup("backend_initialized")
    print("[house7_route] backend initialized", flush=True)
    result = StagedRouteExecutor(backend, output_path=args.output).run(route)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
