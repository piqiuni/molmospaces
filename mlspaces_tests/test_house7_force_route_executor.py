from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "InteractiveNav"
for path in (REPO_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.InteractiveNav.run_house7_force_route import (
    StagedRouteExecutor,
    RosRouteBackend,
    door_exit_goal,
    post_interaction_goals,
    portal_snapshot,
    portal_state_matches,
    route_visualization_plan,
)


class FakeBackend:
    def __init__(self, interaction_success: bool = True) -> None:
        self.calls = []
        self.interaction_success = interaction_success

    def publish_phase(self, payload):
        self.calls.append(("phase", payload["event"]))

    def wait_until_ready(self):
        self.calls.append(("ready",))

    def navigate(self, goal, segment):
        self.calls.append(("navigate", segment, goal))
        return {"success": True, "segment": segment}

    def wait_for_portal_state(self, root, state):
        self.calls.append(("portal", state, root))
        return {"state": state, "room_count": 1 if state == "closed" else 2}

    def interact(self, command):
        self.calls.append(("interact", command["object_id"]))
        return {"success": self.interaction_success, "status": "SUCCEEDED"}

    def wait_for_navigation_map(self):
        self.calls.append(("map_settle",))
        return {"raw_map_updates": 2}

    def latest_graph_summary(self, root):
        return {"state": "open", "room_count": 2, "root": root}


def route_fixture():
    return {
        "route_id": "route_01",
        "door_approach_xyyaw": [1.0, 2.0, 0.0],
        "start_xyyaw": [0.0, 2.0, 0.0],
        "portal_center_xy": [3.0, 2.0],
        "portal_normal_xy": [-1.0, 0.0],
        "far_goal_xyyaw": [6.0, 2.0, 0.0],
        "interaction": {"object_id": "double_door"},
    }


def test_executor_orders_navigation_interaction_and_final_navigation(tmp_path) -> None:
    backend = FakeBackend()
    result = StagedRouteExecutor(backend, tmp_path / "result.json").run(route_fixture())
    assert result["success"] is True
    significant = [call for call in backend.calls if call[0] != "phase"]
    assert significant == [
        ("ready",),
        ("navigate", "approach", [1.0, 2.0, 0.0]),
        ("portal", "closed", "double_door"),
        ("interact", "double_door"),
        ("portal", "open", "double_door"),
        ("map_settle",),
        ("navigate", "crossing", [3.8, 2.0, -0.0]),
        ("map_settle",),
        ("navigate", "final", [6.0, 2.0, 0.0]),
    ]
    assert (tmp_path / "result.json").exists()


def test_executor_does_not_navigate_final_segment_after_interaction_failure() -> None:
    backend = FakeBackend(interaction_success=False)
    result = StagedRouteExecutor(backend).run(route_fixture())
    assert result["success"] is False
    assert not any(call[:2] == ("navigate", "final") for call in backend.calls)


def test_portal_snapshot_matches_source_object_name_and_summarizes_rooms() -> None:
    graph = {
        "graph_revision": 9,
        "nodes": [
            {"id": "room_1", "type": "room"},
            {"id": "room_2", "type": "room"},
            {
                "id": "portal_3",
                "type": "portal",
                "name": "door",
                "attributes": {"source_object_name": "double_door"},
                "interaction": {
                    "state": "open",
                    "traversable": True,
                    "requires_interaction": False,
                    "operation_history": [{"event_id": "open_1"}],
                },
            },
        ],
    }
    snapshot = portal_snapshot(graph, "double_door")
    assert snapshot["state"] == "open"
    assert snapshot["room_count"] == 2
    assert snapshot["graph_revision"] == 9


def test_unknown_blocked_portal_satisfies_closed_route_precondition() -> None:
    snapshot = {
        "state": "unknown",
        "traversable": False,
        "requires_interaction": True,
    }
    assert portal_state_matches(snapshot, "closed") is True
    assert portal_state_matches(snapshot, "open") is False


def test_door_exit_goal_is_on_opposite_side_from_start() -> None:
    goal = door_exit_goal(route_fixture())
    assert goal == [3.8, 2.0, -0.0]


def test_post_interaction_goals_keep_turning_waypoints_and_exclude_final_goal() -> None:
    route = route_fixture()
    route["post_interaction_path_xy"] = [
        [1.0, 2.0],
        [3.8, 2.0],
        [4.5, 2.0],
        [5.0, 2.5],
        [6.0, 2.0],
    ]
    goals = post_interaction_goals(route)
    assert [goal[:2] for goal in goals] == [[3.8, 2.0], [4.5, 2.0], [5.0, 2.5]]


def test_route_visualization_plan_contains_all_goals_and_interaction_goal() -> None:
    route = route_fixture()
    route["post_interaction_path_xy"] = [[4.5, 2.0], [5.0, 2.5]]
    plan = route_visualization_plan(route)
    assert [item["segment"] for item in plan["subgoals"]] == [
        "approach",
        "crossing",
        "post_interaction_01",
        "post_interaction_02",
        "final",
    ]
    assert plan["interaction_goal_xyyaw"] == [1.0, 2.0, 0.0]
    assert plan["interaction_object_id"] == "double_door"


def test_route_backend_defaults_to_four_map_settle_updates() -> None:
    default = inspect.signature(RosRouteBackend.__init__).parameters[
        "map_settle_updates"
    ].default
    assert default == 4
