import math

from semantic_decision_py_pkg.behavior_execution import (
    BehaviorExecutionStateMachine,
    ExecutionConfig,
    NavigationProgressWatchdog,
    STATE_APPROACH_INTERACTION,
    STATE_FINALIZING_EXPLORE,
    STATE_INTERACTING,
    STATE_NAVIGATING,
    STATE_PREPARING_EXPLORE,
    STATE_SUCCEEDED,
    STATE_VERIFYING,
    bounded_empty_plan_retry_delay,
    committed_turn_sign,
    is_post_interaction_traversal_navigation,
    navigation_goal_options,
    navigation_requires_final_yaw,
    navigation_should_prerotate,
    normalize_angle,
    path_lookahead_point,
    requires_graph_verification,
    target_ready_for_graph_verification,
    is_stuck_recovery_failure,
    safe_grid_motion_distance,
)


def test_navigation_progress_watchdog_resets_on_translation_or_rotation() -> None:
    watchdog = NavigationProgressWatchdog(timeout_s=12.0, min_displacement_m=0.10)
    watchdog.reset((0.0, 0.0), now=0.0)
    assert not watchdog.observe((0.01, 0.0), now=11.9)
    assert watchdog.observe((0.01, 0.0), now=12.0)
    watchdog.reset((0.0, 0.0), now=0.0)
    assert not watchdog.observe((0.11, 0.0), now=11.0)
    assert not watchdog.observe((0.12, 0.0), now=22.0)
    watchdog.reset((0.0, 0.0, 0.0), now=0.0)
    assert not watchdog.observe((0.0, 0.0, 0.16), now=11.0)
    assert not watchdog.observe((0.0, 0.0, 0.17), now=22.0)


def test_stuck_recovery_requires_explicit_stagnation_or_oscillation() -> None:
    assert is_stuck_recovery_failure({"reason": "navigation_stagnation"})
    assert is_stuck_recovery_failure({"status": "Robot appears to be oscillating"})
    assert not is_stuck_recovery_failure({"reason": "make_plan_unreachable"})
    assert not is_stuck_recovery_failure({"reason": "final_yaw_alignment_failed"})
    assert not is_stuck_recovery_failure({"status_code": 4, "status": "ABORTED"})


def test_safe_grid_motion_distance_stops_before_rear_obstacle() -> None:
    width = height = 20
    resolution = 0.1
    data = [0] * (width * height)
    data[10 * width + 7] = 100

    safe = safe_grid_motion_distance(
        data,
        width,
        height,
        resolution,
        (0.0, 0.0),
        (1.0, 1.0, 0.0),
        -1.0,
        0.5,
        robot_radius_m=0.1,
        safety_margin_m=0.0,
    )

    assert 0.0 < safe < 0.5


def test_safe_grid_motion_distance_blocks_unknown_space() -> None:
    width = height = 20
    resolution = 0.1
    data = [0] * (width * height)
    data[10 * width + 7] = -1

    assert safe_grid_motion_distance(
        data,
        width,
        height,
        resolution,
        (0.0, 0.0),
        (1.0, 1.0, 0.0),
        -1.0,
        0.5,
        robot_radius_m=0.1,
        safety_margin_m=0.0,
    ) < 0.5


def interaction_candidate(requires_approach=True):
    return {
        "candidate_id": "door_open",
        "behavior_type": "INTERACT",
        "goal_xyyaw": [1.0, 2.0, 0.0],
        "interaction_command": {"expected_state": "open"},
        "metadata": {"requires_approach": requires_approach},
    }


def test_committed_turn_sign_is_stable_at_pi_boundary() -> None:
    assert committed_turn_sign(math.pi - 0.05) == -1
    assert committed_turn_sign(-math.pi + 0.05) == -1
    assert committed_turn_sign(1.0) == 1
    assert committed_turn_sign(-1.0) == -1
    assert math.isclose(normalize_angle(3.0 * math.pi), math.pi, abs_tol=1e-6)


def test_path_lookahead_uses_plan_direction_instead_of_final_goal_bearing() -> None:
    lookahead = path_lookahead_point(
        (0.0, 0.0),
        [(0.0, 0.0), (-0.4, 0.0), (-0.8, 0.1), (1.0, 2.0)],
        0.7,
    )

    assert lookahead == (-0.8, 0.1)


def test_navigation_goal_options_preserve_nearest_first_and_remove_duplicates() -> None:
    candidate = {
        "goal_xyyaw": [1.0, 2.0, 0.0],
        "metadata": {
            "goal_xyyaw_candidates": [
                [1.0, 2.0, 0.0],
                [1.25, 2.0, 0.0],
                [1.50, 2.0, math.pi],
            ]
        },
    }

    assert navigation_goal_options(candidate) == [
        (1.0, 2.0, 0.0),
        (1.25, 2.0, 0.0),
        (1.50, 2.0, math.pi),
    ]


def test_explore_navigation_skips_prerotation_and_final_yaw_alignment() -> None:
    assert not navigation_should_prerotate("EXPLORE")
    assert navigation_should_prerotate("INTERACT")
    assert navigation_should_prerotate("NAVIGATE")

    assert not navigation_requires_final_yaw("EXPLORE", True, [1.0, 2.0, 0.5])
    assert navigation_requires_final_yaw("INTERACT", True, [1.0, 2.0, 0.5])
    assert navigation_requires_final_yaw("NAVIGATE", True, [1.0, 2.0, 0.5])
    assert not navigation_requires_final_yaw("INTERACT", False, [1.0, 2.0, 0.5])
    assert not navigation_requires_final_yaw("INTERACT", True, [1.0, 2.0])


def test_post_interaction_traversal_retry_is_scoped_to_navigate_continuation() -> None:
    traversal = {
        "behavior_type": "NAVIGATE",
        "metadata": {"post_interaction_traversal": True},
    }
    assert is_post_interaction_traversal_navigation(traversal)
    assert not is_post_interaction_traversal_navigation(
        {"behavior_type": "NAVIGATE", "metadata": {}}
    )
    assert not is_post_interaction_traversal_navigation(
        {"behavior_type": "INTERACT", "metadata": traversal["metadata"]}
    )
    assert not is_post_interaction_traversal_navigation(
        {"behavior_type": "EXPLORE", "metadata": traversal["metadata"]}
    )


def test_empty_plan_retry_delay_is_bounded_by_deadline() -> None:
    assert bounded_empty_plan_retry_delay(10.0, 18.0, 0.5) == 0.5
    assert math.isclose(
        bounded_empty_plan_retry_delay(17.8, 18.0, 0.5),
        0.2,
        abs_tol=1e-9,
    )
    assert bounded_empty_plan_retry_delay(18.0, 18.0, 0.5) is None
    assert bounded_empty_plan_retry_delay(18.1, 18.0, 0.5) is None


def test_target_navigation_uses_graph_verification_for_mllm_module3() -> None:
    assert requires_graph_verification("mllm_skill_verified", target_candidate())
    assert not requires_graph_verification(
        "mllm_skill_verified",
        {"behavior_type": "NAVIGATE", "metadata": {"target_goal": False}},
    )
    assert requires_graph_verification("rule_verified", interaction_candidate())


def target_candidate():
    return {
        "candidate_id": "target:fridge_1",
        "behavior_type": "NAVIGATE",
        "target_id": "fridge_1",
        "goal_xyyaw": [4.0, 2.0, 0.0],
        "metadata": {
            "target_goal": True,
            "verify_target_visibility": True,
            "target_min_visible_pixels": 16,
        },
    }


def test_interaction_execution_orders_approach_action_and_verification() -> None:
    machine = BehaviorExecutionStateMachine()
    commands = machine.start(interaction_candidate(), now=0.0)
    assert commands[0]["kind"] == "navigate"
    commands = machine.on_navigation_result(True, now=1.0)
    assert machine.state == STATE_INTERACTING
    assert commands[0]["kind"] == "interact"
    assert machine.on_interaction_result(True, now=2.0) == []
    assert machine.state == STATE_VERIFYING
    assert machine.on_graph_state("closed", now=3.0) == []
    terminal = machine.on_graph_state("open", now=4.0)
    assert machine.state == STATE_SUCCEEDED
    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is True


def test_explore_behavior_reserves_navigates_and_finalizes() -> None:
    machine = BehaviorExecutionStateMachine()
    commands = machine.start(
        {
            "candidate_id": "frontier_1",
            "behavior_type": "EXPLORE",
            "goal_xyyaw": [9.0, 9.0, 0.0],
            "metadata": {"cluster_id": "cluster_1"},
        },
        now=0.0,
    )
    assert machine.state == STATE_PREPARING_EXPLORE
    assert commands[0]["kind"] == "reserve_frontier"

    commands = machine.on_explore_ready(
        {"goal_xyyaw": [1.0, 2.0, 0.75], "frame_id": "map"}, now=1.0
    )
    assert machine.state == STATE_NAVIGATING
    assert commands[0]["kind"] == "navigate"
    assert commands[0]["candidate"]["goal_xyyaw"] == [1.0, 2.0, 0.75]
    assert commands[0]["candidate"]["metadata"]["frame_id"] == "map"

    commands = machine.on_navigation_result(True, {"status": "SUCCEEDED"}, now=2.0)
    assert machine.state == STATE_FINALIZING_EXPLORE
    assert commands[0]["kind"] == "finalize_frontier"
    assert commands[0]["success"] is True

    terminal = machine.on_explore_result(True, {"event": "frontier_gone"}, now=3.0)
    assert terminal[0]["success"] is True


def test_explore_reservation_failure_finishes_without_navigation() -> None:
    machine = BehaviorExecutionStateMachine()
    machine.start(
        {
            "candidate_id": "frontier_1",
            "behavior_type": "EXPLORE",
            "metadata": {"cluster_id": "cluster_1"},
        },
        now=0.0,
    )
    terminal = machine.on_explore_result(
        False, {"reason": "frontier_candidate_not_available"}, now=1.0
    )
    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is False


def test_explore_ignores_terminal_feedback_until_finalization() -> None:
    machine = BehaviorExecutionStateMachine()
    machine.start(
        {
            "candidate_id": "frontier_1",
            "behavior_type": "EXPLORE",
            "metadata": {"cluster_id": "cluster_1"},
        },
        now=0.0,
    )
    machine.on_explore_ready({"goal_xyyaw": [1.0, 2.0, 0.0]}, now=1.0)

    assert machine.on_explore_result(False, {"reason": "stale_move_base_status"}, now=2.0) == []
    assert machine.state == STATE_NAVIGATING


def test_explore_navigation_timeout_requests_frontier_finalization() -> None:
    machine = BehaviorExecutionStateMachine(
        ExecutionConfig(navigation_timeout_s=2.0)
    )
    machine.start(
        {
            "candidate_id": "frontier_1",
            "behavior_type": "EXPLORE",
            "metadata": {"cluster_id": "cluster_1"},
        },
        now=0.0,
    )
    machine.on_explore_ready({"goal_xyyaw": [1.0, 2.0, 0.0]}, now=1.0)
    assert machine.timeout_reason(now=3.1) == "navigation_timeout"
    commands = machine.fail_timeout("navigation_timeout", now=3.1)
    assert machine.state == STATE_FINALIZING_EXPLORE
    assert commands[0]["kind"] == "finalize_frontier"
    assert commands[0]["success"] is False


def test_interaction_approach_uses_short_navigation_timeout() -> None:
    machine = BehaviorExecutionStateMachine(
        ExecutionConfig(
            navigation_timeout_s=180.0,
            interaction_navigation_timeout_s=2.0,
        )
    )
    machine.start(interaction_candidate(), now=0.0)
    assert machine.state == STATE_APPROACH_INTERACTION
    assert machine.timeout_reason(now=2.1) == "interaction_navigation_timeout"


def test_target_navigation_waits_for_visibility_verification() -> None:
    machine = BehaviorExecutionStateMachine()
    commands = machine.start(target_candidate(), now=0.0)
    assert commands[0]["kind"] == "navigate"

    assert machine.on_navigation_result(True, now=1.0) == []
    assert machine.state == STATE_VERIFYING
    assert machine.on_target_visibility(False, now=2.0) == []
    terminal = machine.on_target_visibility(
        True,
        detail={"visible_pixels": 24, "min_visible_pixels": 16},
        now=3.0,
    )

    assert machine.state == STATE_SUCCEEDED
    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is True


def test_visible_reliable_target_skips_navigation_and_verifies_graph() -> None:
    candidate = target_candidate()
    candidate["metadata"].update(
        {
            "target_visible_now": True,
            "target_reliably_observed": True,
            "target_navigation_required": False,
        }
    )
    assert target_ready_for_graph_verification(candidate)

    machine = BehaviorExecutionStateMachine()
    assert machine.start(candidate, now=0.0) == []
    assert machine.state == STATE_VERIFYING

    terminal = machine.on_target_visibility(
        True,
        detail={"visible_pixels": 24, "min_visible_pixels": 16},
        now=0.1,
    )
    assert machine.state == STATE_SUCCEEDED
    assert terminal[0]["success"] is True


def test_visible_reliable_target_still_navigates_when_goal_pose_is_not_reached() -> None:
    candidate = target_candidate()
    candidate["metadata"].update(
        {
            "target_visible_now": True,
            "target_reliably_observed": True,
            "target_navigation_required": True,
        }
    )

    assert not target_ready_for_graph_verification(candidate)
    machine = BehaviorExecutionStateMachine()
    commands = machine.start(candidate, now=0.0)
    assert machine.state == STATE_NAVIGATING
    assert commands[0]["kind"] == "navigate"


def test_historical_target_observation_does_not_skip_navigation() -> None:
    candidate = target_candidate()
    candidate["metadata"].update(
        {
            "target_visible_now": False,
            "target_reliably_observed": True,
        }
    )
    assert not target_ready_for_graph_verification(candidate)

    machine = BehaviorExecutionStateMachine()
    commands = machine.start(candidate, now=0.0)
    assert machine.state == STATE_NAVIGATING
    assert commands[0]["kind"] == "navigate"


def test_navigation_without_visibility_requirement_finishes_immediately() -> None:
    candidate = target_candidate()
    candidate["metadata"]["verify_target_visibility"] = False
    machine = BehaviorExecutionStateMachine()
    machine.start(candidate, now=0.0)

    terminal = machine.on_navigation_result(True, now=1.0)

    assert machine.state == STATE_SUCCEEDED
    assert terminal[0]["success"] is True
