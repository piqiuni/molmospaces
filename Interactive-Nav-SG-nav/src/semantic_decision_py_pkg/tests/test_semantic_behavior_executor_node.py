from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import rospy
from std_msgs.msg import String


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


import semantic_behavior_executor as executor_module
from semantic_behavior_executor import SemanticBehaviorExecutor
from semantic_decision_py_pkg.behavior_execution import (
    BehaviorExecutionStateMachine,
    STATE_IDLE,
    STATE_INTERACTING,
    STATE_NAVIGATING,
    STATE_SUCCEEDED,
    STATE_VERIFYING,
)


class FakePublisher:
    def __init__(self, error: Exception | None = None) -> None:
        self.messages = []
        self.error = error

    def publish(self, message) -> None:
        if self.error is not None:
            raise self.error
        self.messages.append(message)


class FakeMoveBase:
    def __init__(self) -> None:
        self.cancel_count = 0
        self.wait_count = 0
        self.on_wait = None
        self.state = executor_module.GoalStatus.LOST
        self.cancel_sets_terminal = True
        self.server_available = False
        self.states_after_send = []
        self.sent_goals = []
        self.status_text = ""

    def cancel_goal(self) -> None:
        self.cancel_count += 1
        if self.cancel_sets_terminal:
            self.state = executor_module.GoalStatus.PREEMPTED

    def get_state(self) -> int:
        return self.state

    def get_goal_status_text(self) -> str:
        return self.status_text

    def send_goal(self, goal) -> None:
        pose = goal.target_pose.pose
        self.sent_goals.append((pose.position.x, pose.position.y))
        if self.states_after_send:
            self.state = self.states_after_send.pop(0)

    def wait_for_server(self, _duration) -> bool:
        self.wait_count += 1
        if self.on_wait is not None:
            self.on_wait()
        return self.server_available


def make_node(*, episode_active: bool = True, episode_generation: int = 7):
    node = object.__new__(SemanticBehaviorExecutor)
    node.lock = threading.RLock()
    node.episode_active = episode_active
    node.episode_generation = episode_generation
    node.selection = None
    node.machine = BehaviorExecutionStateMachine()
    node.ablation = SimpleNamespace(module3="direct_atomic")
    node.feedback_pub = FakePublisher()
    node.cmd_vel_pub = FakePublisher()
    node.move_base = FakeMoveBase()
    node.active_skill_plan = {}
    node.pending_skill_actions = []
    node.interaction_command_sequence = 0
    node.verification_retries = 0
    node.model_events = []
    node.navigation_cancel_wait_s = 1.0
    node.target_navigation_stagnation_timeout_s = 4.0
    node.navigation_stagnation_timeout_s = 12.0
    node.navigation_stagnation_distance_m = 0.10
    node.navigation_stagnation_yaw_rad = 0.15
    node.make_plan_empty_retry_count = 0
    node.make_plan_empty_retry_delay_s = 0.0
    node.explore_make_plan_fail_open_after_retries = False
    node.final_align_enabled = False
    node.map_frame = "map"
    node._current_pose = lambda _frame_id: (0.0, 0.0, 0.0)
    node._last_explore_reservation_publish_at = 0.0
    node._explore_reservation_publish_count = 0
    node._explore_feedback_received_count = 0
    node._explore_feedback_matched_count = 0
    node._explore_feedback_ignored_count = 0
    node._last_explore_feedback = {}
    node._target_visibility_retry = None
    node.target_visibility_retry_delay_s = 1.0
    node.latest_graph = {}
    node.latest_image_sequence = 0
    node.pre_interaction_image_sequence = 0
    node._stuck_failure_origin_xy = None
    node._stuck_failure_candidate_ids = set()
    node.dispatched = []
    node._dispatch = lambda commands: node.dispatched.extend(commands)
    return node


def navigation_selection(generation: int = 7) -> dict:
    return {
        "decision_id": "decision_000007",
        "candidate_id": "target:television",
        "behavior_type": "NAVIGATE",
        "goal_xyyaw": [1.0, 2.0, 0.0],
        "episode_generation": generation,
        "metadata": {},
    }


def select(node, selection: dict) -> None:
    node._selection_callback(String(data=json.dumps(selection)))


def test_selection_is_ignored_when_episode_is_inactive_or_generation_is_stale() -> None:
    inactive = make_node(episode_active=False)
    select(inactive, navigation_selection())
    assert inactive.selection is None
    assert inactive.feedback_pub.messages == []

    stale = make_node(episode_generation=8)
    select(stale, navigation_selection(generation=7))
    assert stale.selection is None
    assert stale.feedback_pub.messages == []

    current = make_node(episode_generation=8)
    select(current, navigation_selection(generation=8))
    assert current.selection["decision_id"] == "decision_000007"
    assert current.machine.state == STATE_NAVIGATING
    assert [command["kind"] for command in current.dispatched] == ["navigate"]
    feedback = json.loads(current.feedback_pub.messages[-1].data)
    assert feedback["status"] == "STARTED"
    assert feedback["episode_generation"] == 8


def test_inactive_target_context_clears_execution_and_stops_motion(monkeypatch) -> None:
    monkeypatch.setattr(executor_module.rospy, "is_shutdown", lambda: False)
    node = make_node()
    select(node, navigation_selection())
    node.active_skill_plan = {"operation": "open"}
    node.pending_skill_actions = [{"skill": "pull"}]
    node.model_events = [{"role": "skill_planning"}]
    node._explore_feedback_received_count = 3

    node._target_context_callback(
        String(
            data=json.dumps(
                {
                    "target_context": {
                        "episode_active": False,
                        "episode_generation": 7,
                    }
                }
            )
        )
    )

    assert node.episode_active is False
    assert node.selection is None
    assert node.machine.state == STATE_IDLE
    assert node.active_skill_plan == {}
    assert node.pending_skill_actions == []
    assert node.model_events == []
    assert node._explore_feedback_received_count == 0
    assert node.move_base.cancel_count == 1
    assert len(node.cmd_vel_pub.messages) == 1
    assert node.cmd_vel_pub.messages[0].linear.x == 0.0
    assert node.cmd_vel_pub.messages[0].angular.z == 0.0


def test_generation_change_invalidates_old_execution_even_when_active(monkeypatch) -> None:
    monkeypatch.setattr(executor_module.rospy, "is_shutdown", lambda: False)
    node = make_node(episode_generation=7)
    select(node, navigation_selection(generation=7))

    node._target_context_callback(
        String(data=json.dumps({"episode_active": True, "episode_generation": 8}))
    )

    assert node.episode_active is True
    assert node.episode_generation == 8
    assert node.selection is None
    assert node.machine.state == STATE_IDLE
    assert node.move_base.cancel_count == 1


def test_navigation_and_interaction_current_checks_include_lifecycle() -> None:
    node = make_node(episode_generation=7)
    node.selection = navigation_selection(generation=7)
    node.machine.state = STATE_NAVIGATING
    assert node._navigation_is_current("decision_000007") is True

    node.episode_generation = 8
    assert node._navigation_is_current("decision_000007") is False

    node.episode_generation = 7
    node.machine.state = STATE_INTERACTING
    assert node._interaction_is_current("decision_000007") is True
    node.episode_active = False
    assert node._interaction_is_current("decision_000007") is False


def test_feedback_generation_is_checked_only_when_backend_echoes_it() -> None:
    node = make_node(episode_generation=7)
    node.selection = navigation_selection(generation=7)

    identity = {
        "command_id": node._command_id(node.selection),
        "candidate_id": node.selection["candidate_id"],
    }
    assert node._matches_active(identity) is True
    assert node._matches_active({**identity, "episode_generation": 6}) is False
    assert node._matches_active({**identity, "episode_generation": 7}) is True


def test_cmd_vel_publish_swallows_shutdown_ros_exception(monkeypatch) -> None:
    monkeypatch.setattr(executor_module.rospy, "is_shutdown", lambda: False)
    node = make_node()
    node.cmd_vel_pub = FakePublisher(rospy.ROSException("closed topic"))

    assert node._publish_cmd_vel_safely(executor_module.Twist()) is False


def test_wait_for_move_base_server_stops_when_lifecycle_changes(monkeypatch) -> None:
    monkeypatch.setattr(executor_module.rospy, "is_shutdown", lambda: False)
    node = make_node()
    node.selection = navigation_selection()
    node.machine.state = STATE_NAVIGATING
    node.move_base.on_wait = lambda: setattr(node, "episode_active", False)

    assert node._wait_for_move_base_server("decision_000007", timeout_s=30.0) is False
    assert node.move_base.wait_count == 1


def test_navigation_waits_for_stale_move_base_goal_to_be_canceled(monkeypatch) -> None:
    monkeypatch.setattr(executor_module.rospy, "is_shutdown", lambda: False)
    node = make_node()
    node.selection = navigation_selection()
    node.machine.state = STATE_NAVIGATING
    node.move_base.state = executor_module.GoalStatus.ACTIVE

    assert node._wait_for_move_base_inactive("decision_000007") is True
    assert node.move_base.cancel_count == 1
    assert node.move_base.state == executor_module.GoalStatus.PREEMPTED


def _configure_runtime_navigation(node, monkeypatch):
    monkeypatch.setattr(
        executor_module.rospy.Time,
        "now",
        lambda: executor_module.rospy.Time(0),
    )
    node.move_base.server_available = True
    node._wait_for_move_base_inactive = lambda _decision_id: True
    node._preflight_navigation_plan = lambda *_args: (True, None, "reachable")
    node._prerotate_for_rear_goal = lambda *_args, **_kwargs: True
    node._current_pose = lambda _frame_id: (0.0, 0.0, 0.0)
    results = []
    node._handle_navigation_result = (
        lambda _decision_id, success, detail: results.append((success, detail))
    )
    return results


def _target_navigation_with_fallback() -> dict:
    return {
        **navigation_selection(),
        "metadata": {
            "target_goal": True,
            "goal_xyyaw_candidates": [[3.0, 4.0, 0.0]],
        },
    }


def test_target_navigation_retries_next_pose_after_move_base_aborts(monkeypatch) -> None:
    monkeypatch.setattr(executor_module.rospy, "is_shutdown", lambda: False)
    node = make_node()
    candidate = _target_navigation_with_fallback()
    node.selection = candidate
    node.machine.state = STATE_NAVIGATING
    results = _configure_runtime_navigation(node, monkeypatch)
    node.move_base.states_after_send = [
        executor_module.GoalStatus.ABORTED,
        executor_module.GoalStatus.SUCCEEDED,
    ]

    node._run_navigation(candidate["decision_id"], candidate)

    assert node.move_base.sent_goals == [(1.0, 2.0), (3.0, 4.0)]
    assert len(results) == 1
    assert results[0][0] is True
    detail = results[0][1]
    assert detail["runtime_fallback_count"] == 1
    assert [attempt["execution_status_code"] for attempt in detail["attempted_goals"]] == [
        executor_module.GoalStatus.ABORTED,
        executor_module.GoalStatus.SUCCEEDED,
    ]


def test_target_visibility_retry_starts_at_the_next_untried_goal(monkeypatch) -> None:
    monkeypatch.setattr(executor_module.rospy, "is_shutdown", lambda: False)
    node = make_node()
    candidate = {
        **navigation_selection(),
        "metadata": {
            "target_goal": True,
            "verify_target_visibility": True,
            "goal_xyyaw_candidates": [
                [1.0, 2.0, 0.0],
                [3.0, 4.0, 0.0],
                [5.0, 6.0, 0.0],
            ],
        },
    }
    node.selection = candidate
    node.machine.state = STATE_NAVIGATING
    results = _configure_runtime_navigation(node, monkeypatch)
    node.move_base.states_after_send = [executor_module.GoalStatus.SUCCEEDED]

    node._run_navigation(candidate["decision_id"], candidate, start_goal_index=2)

    assert node.move_base.sent_goals == [(5.0, 6.0)]
    assert len(results) == 1
    assert results[0][0] is True
    assert results[0][1]["executed_goal_index"] == 2


def test_target_visibility_retry_keeps_active_decision_and_uses_next_goal() -> None:
    node = make_node()
    candidate = {
        **navigation_selection(),
        "metadata": {
            "target_goal": True,
            "verify_target_visibility": True,
            "goal_xyyaw_candidates": [
                [1.0, 2.0, 0.0],
                [3.0, 4.0, 0.0],
                [5.0, 6.0, 0.0],
            ],
        },
    }
    node.selection = candidate
    node.machine.candidate = dict(candidate)
    node.machine.state = STATE_VERIFYING
    node._target_visibility_retry = {
        "decision_id": candidate["decision_id"],
        "next_goal_option_index": 2,
        "ready_at_monotonic": 0.0,
        "reason": "target_visibility_unconfirmed",
    }

    commands = node._advance_target_visibility_retry_locked(now=1.0)

    assert node.selection is candidate
    assert node.machine.state == STATE_NAVIGATING
    assert node._target_visibility_retry is None
    assert [command["kind"] for command in commands] == ["navigate"]
    assert commands[0]["start_goal_index"] == 2


def test_target_visibility_retry_defers_stale_graph_completion_until_grace_period() -> None:
    node = make_node()
    candidate = {
        **navigation_selection(),
        "target_id": "plant_1",
        "target_name": "plant",
        "metadata": {
            "target_goal": True,
            "verify_target_visibility": True,
            "goal_xyyaw_candidates": [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]],
            "target_success_distance_m": 1.5,
        },
    }
    node.selection = candidate
    node.machine.candidate = dict(candidate)
    node.machine.state = STATE_VERIFYING
    node.latest_graph = {
        "graph_revision": 12,
        "nodes": [
            {
                "id": "plant_1",
                "aabb_center": [1.0, 0.0, 0.5],
                "is_currently_visible": True,
                "attributes": {
                    "visible_pixels": 76,
                    "visible_fraction": 0.5,
                    "consecutive_observations": 2,
                },
            }
        ],
    }
    node._target_visibility_retry = {
        "decision_id": candidate["decision_id"],
        "next_goal_option_index": 1,
        "ready_at_monotonic": 5.0,
        "reason": "target_visibility_unconfirmed",
    }

    assert node._verify_graph_locked() == []
    assert node.machine.state == STATE_VERIFYING

    commands = node._advance_target_visibility_retry_locked(now=6.0)

    assert commands and commands[-1]["kind"] == "terminal"
    assert commands[-1]["success"] is True
    assert node.machine.state == STATE_SUCCEEDED
    assert node._target_visibility_retry is None


def test_target_navigation_does_not_retry_after_preemption(monkeypatch) -> None:
    monkeypatch.setattr(executor_module.rospy, "is_shutdown", lambda: False)
    node = make_node()
    candidate = _target_navigation_with_fallback()
    node.selection = candidate
    node.machine.state = STATE_NAVIGATING
    results = _configure_runtime_navigation(node, monkeypatch)
    node.move_base.states_after_send = [
        executor_module.GoalStatus.PREEMPTED,
        executor_module.GoalStatus.SUCCEEDED,
    ]

    node._run_navigation(candidate["decision_id"], candidate)

    assert node.move_base.sent_goals == [(1.0, 2.0)]
    assert len(results) == 1
    assert results[0][0] is False
    assert results[0][1]["status_code"] == executor_module.GoalStatus.PREEMPTED


def test_target_navigation_retries_next_pose_after_stagnation(monkeypatch) -> None:
    class ImmediateStagnationWatchdog:
        timeout_values = []

        def __init__(self, *, timeout_s, **_kwargs) -> None:
            self.timeout_values.append(timeout_s)

        def reset(self, _pose, _now) -> None:
            pass

        def observe(self, _pose, _now) -> bool:
            return True

    monkeypatch.setattr(executor_module.rospy, "is_shutdown", lambda: False)
    monkeypatch.setattr(
        executor_module,
        "NavigationProgressWatchdog",
        ImmediateStagnationWatchdog,
    )
    node = make_node()
    candidate = _target_navigation_with_fallback()
    node.selection = candidate
    node.machine.state = STATE_NAVIGATING
    results = _configure_runtime_navigation(node, monkeypatch)
    node.move_base.states_after_send = [
        executor_module.GoalStatus.ACTIVE,
        executor_module.GoalStatus.SUCCEEDED,
    ]

    node._run_navigation(candidate["decision_id"], candidate)

    assert node.move_base.sent_goals == [(1.0, 2.0), (3.0, 4.0)]
    assert node.move_base.cancel_count == 1
    assert ImmediateStagnationWatchdog.timeout_values == [4.0, 4.0]
    assert len(results) == 1
    assert results[0][0] is True
    assert results[0][1]["runtime_fallback_count"] == 1


def test_target_verification_preserves_zero_visible_fraction_threshold() -> None:
    node = make_node()
    node.selection = {
        **navigation_selection(),
        "target_id": "plant_1",
        "target_name": "plant",
        "metadata": {
            "target_goal": True,
            "target_min_visible_pixels": 16,
            "target_min_visible_fraction": 0.0,
            "target_min_consecutive_observations": 1,
            "target_success_distance_m": 1.5,
        },
    }
    node.machine.candidate = dict(node.selection)
    node.machine.state = STATE_VERIFYING
    node.latest_graph = {
        "graph_revision": 12,
        "nodes": [
            {
                "id": "plant_1",
                "aabb_center": [1.0, 0.0, 0.5],
                "is_currently_visible": True,
                "attributes": {
                    "visible_pixels": 76,
                    "visible_fraction": 0.006,
                    "consecutive_observations": 1,
                },
            }
        ],
    }

    commands = node._verify_graph_locked()

    assert commands and commands[-1]["kind"] == "terminal"
    assert node.machine.state == STATE_SUCCEEDED


def test_target_verification_requires_distance_inside_the_success_threshold() -> None:
    node = make_node()
    node.selection = {
        **navigation_selection(),
        "target_id": "plant_1",
        "target_name": "plant",
        "metadata": {
            "target_goal": True,
            "target_min_visible_pixels": 16,
            "target_min_visible_fraction": 0.0,
            "target_min_consecutive_observations": 1,
            "target_success_distance_m": 1.5,
        },
    }
    node.machine.candidate = dict(node.selection)
    node.machine.state = STATE_VERIFYING
    node.latest_graph = {
        "graph_revision": 12,
        "nodes": [
            {
                "id": "plant_1",
                # Equal to the official threshold is not success: NavToObj
                # requires the distance to be strictly inside the threshold.
                "aabb_center": [1.5, 0.0, 0.5],
                "is_currently_visible": True,
                "attributes": {
                    "visible_pixels": 76,
                    "visible_fraction": 0.006,
                    "consecutive_observations": 1,
                },
            }
        ],
    }

    commands = node._verify_graph_locked()

    assert commands == []
    assert node.machine.state == STATE_VERIFYING


def test_target_verification_does_not_accept_same_name_distractor_without_id() -> None:
    node = make_node()
    node.selection = {
        **navigation_selection(),
        "target_id": "plant_selected",
        "target_name": "plant",
        "metadata": {
            "target_goal": True,
            "target_min_visible_pixels": 16,
            "target_min_visible_fraction": 0.0,
            "target_min_consecutive_observations": 1,
        },
    }
    node.machine.candidate = dict(node.selection)
    node.machine.state = STATE_VERIFYING
    node.latest_graph = {
        "graph_revision": 12,
        "nodes": [
            {
                "id": "plant_distractor",
                "is_currently_visible": True,
                "attributes": {
                    "source_object_name": "plant",
                    "visible_pixels": 76,
                    "visible_fraction": 1.0,
                    "consecutive_observations": 1,
                },
            }
        ],
    }

    commands = node._verify_graph_locked()

    assert commands == []
    assert node.machine.state == STATE_VERIFYING
