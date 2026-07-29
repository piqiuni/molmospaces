import math

from semantic_decision_py_pkg.behavior_execution import (
    BehaviorExecutionStateMachine,
    ExecutionConfig,
    NavigationProgressWatchdog,
    PostInteractionCostmapBaseline,
    PostInteractionRawMapBarrier,
    STATE_APPROACH_INTERACTION,
    STATE_FINALIZING_EXPLORE,
    STATE_INTERACTING,
    STATE_NAVIGATING,
    STATE_PREPARING_EXPLORE,
    STATE_SUCCEEDED,
    STATE_WAITING_FOR_DRAWER_SCAN,
    STATE_VERIFYING,
    bounded_empty_plan_retry_delay,
    committed_turn_sign,
    is_post_interaction_traversal_navigation,
    navigation_goal_options,
    navigation_prerotation_heading_target,
    navigation_requires_final_yaw,
    navigation_should_prerotate,
    next_interaction_approach_option_index,
    normalize_angle,
    path_lookahead_point,
    post_open_path_is_confirmed,
    post_open_path_retryable_preflight_reason,
    post_interaction_costmap_baseline_keys,
    post_interaction_costmap_receipts_fresh_source,
    post_interaction_costmap_fresh_source,
    post_interaction_costmap_is_fresh,
    post_interaction_planning_occupancy_fresh_source,
    post_interaction_raw_occupancy_fresh_source,
    prerotation_control_step_budget,
    prerotation_rgb_step_gate,
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


def test_navigation_progress_watchdog_keeps_a_fresh_local_plan_making_goal_progress() -> None:
    watchdog = NavigationProgressWatchdog(
        timeout_s=12.0,
        min_displacement_m=0.10,
        min_goal_distance_reduction_m=0.02,
    )
    watchdog.reset((0.0, 0.0, 0.0), now=0.0, goal_distance_m=2.0)

    # The base has moved less than 10 cm, but an active local plan has made a
    # real 3 cm reduction toward the selected goal.  This must not become a
    # false semantic "stagnation" cancellation.
    assert not watchdog.observe(
        (0.03, 0.0, 0.0),
        now=12.0,
        goal_distance_m=1.97,
        local_plan_fresh=True,
    )
    assert not watchdog.observe(
        (0.03, 0.0, 0.0),
        now=23.9,
        goal_distance_m=1.97,
        local_plan_fresh=False,
    )
    assert watchdog.observe(
        (0.03, 0.0, 0.0),
        now=24.0,
        goal_distance_m=1.97,
        local_plan_fresh=False,
    )


def test_stuck_recovery_accepts_repeated_no_progress_plan_failures() -> None:
    assert is_stuck_recovery_failure({"reason": "navigation_stagnation"})
    assert is_stuck_recovery_failure({"status": "Robot appears to be oscillating"})
    assert is_stuck_recovery_failure({"reason": "make_plan_unreachable"})
    assert is_stuck_recovery_failure({"status": "Failed to get a plan"})
    assert not is_stuck_recovery_failure({"reason": "final_yaw_alignment_failed"})
    assert not is_stuck_recovery_failure({"status_code": 4, "status": "ABORTED"})


def test_interaction_approach_fallback_is_stagnation_only_and_bounded() -> None:
    kwargs = {
        "behavior_type": "INTERACT",
        "failure_detail": {"reason": "navigation_stagnation"},
        "selected_option_index": 0,
        "attempted_navigation_count": 1,
        "max_navigation_attempts": 3,
        "goal_option_count": 4,
    }
    assert next_interaction_approach_option_index(**kwargs) == 1
    assert (
        next_interaction_approach_option_index(
            **{**kwargs, "attempted_navigation_count": 3}
        )
        is None
    )
    assert (
        next_interaction_approach_option_index(
            **{**kwargs, "failure_detail": {"reason": "make_plan_unreachable"}}
        )
        is None
    )
    assert (
        next_interaction_approach_option_index(
            **{**kwargs, "behavior_type": "NAVIGATE"}
        )
        is None
    )


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


def test_prerotation_step_budget_uses_only_required_v3_control_steps() -> None:
    assert prerotation_control_step_budget(
        math.pi,
        math.pi / 6.0,
        speed_rad_s=1.25,
        control_dt_s=0.2,
        max_control_steps=12,
    ) == 11
    assert prerotation_control_step_budget(
        math.pi / 6.0 + 0.01,
        math.pi / 6.0,
        speed_rad_s=1.25,
        control_dt_s=0.2,
        max_control_steps=12,
    ) == 1
    assert prerotation_control_step_budget(
        0.1,
        math.pi / 6.0,
        speed_rad_s=1.25,
        control_dt_s=0.2,
        max_control_steps=12,
    ) == 0


def test_prerotation_rgb_step_gate_allows_one_command_per_evaluator_step() -> None:
    assert prerotation_rgb_step_gate(
        last_sent_rgb_step_seq=10,
        current_rgb_step_seq=10,
        nonzero_commands_sent=0,
        max_control_steps=12,
    ) == "wait"
    # A jump is still one new eligible command, not one per skipped sequence.
    assert prerotation_rgb_step_gate(
        last_sent_rgb_step_seq=10,
        current_rgb_step_seq=14,
        nonzero_commands_sent=1,
        max_control_steps=12,
    ) == "send"
    assert prerotation_rgb_step_gate(
        last_sent_rgb_step_seq=14,
        current_rgb_step_seq=13,
        nonzero_commands_sent=2,
        max_control_steps=12,
    ) == "stop"
    assert prerotation_rgb_step_gate(
        last_sent_rgb_step_seq=14,
        current_rgb_step_seq=15,
        nonzero_commands_sent=12,
        max_control_steps=12,
    ) == "stop"
    assert prerotation_rgb_step_gate(
        last_sent_rgb_step_seq=14,
        current_rgb_step_seq=None,
        nonzero_commands_sent=2,
        max_control_steps=12,
    ) == "wait"


def test_path_lookahead_uses_plan_direction_instead_of_final_goal_bearing() -> None:
    lookahead = path_lookahead_point(
        (0.0, 0.0),
        [(0.0, 0.0), (-0.4, 0.0), (-0.8, 0.1), (1.0, 2.0)],
        0.7,
    )

    assert lookahead == (-0.8, 0.1)


def test_navigation_prerotation_never_falls_back_to_final_goal() -> None:
    assert navigation_prerotation_heading_target(None) is None
    assert navigation_prerotation_heading_target((-0.8, 0.1)) == (-0.8, 0.1)


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


def test_post_open_costmap_gate_requires_a_new_receipt_after_open() -> None:
    baseline = PostInteractionCostmapBaseline(
        portal_id="portal_47",
        source_event_id="object_skill_000047",
        receipt_count=18,
        header_seq=941,
        update_receipt_count=41,
        update_header_seq=502,
    )

    # Header values are diagnostic only: local receipt counters must advance.
    # Incremental costmap_updates are primary; sparse full-grid publications
    # remain a valid fallback.
    assert not post_interaction_costmap_is_fresh(baseline, 18, 41)
    assert post_interaction_costmap_fresh_source(baseline, 19, 41) == "full"
    assert post_interaction_costmap_is_fresh(baseline, 19, 41)
    assert (
        post_interaction_costmap_fresh_source(baseline, 18, 42)
        == "costmap_update"
    )
    assert (
        post_interaction_costmap_fresh_source(baseline, 19, 42)
        == "costmap_update"
    )


def test_post_open_causal_map_gate_requires_raw_then_planning_then_costmap() -> None:
    baseline = PostInteractionCostmapBaseline(
        portal_id="portal_47",
        source_event_id="object_skill_000047",
        receipt_count=18,
        update_receipt_count=41,
        raw_occupancy_receipt_count=7,
        planning_occupancy_receipt_count=12,
        interaction_result_stamp_sec=100.0,
    )

    # A raw map received after the result but stamped before it is precisely
    # the residual-map failure this gate must reject.
    assert post_interaction_raw_occupancy_fresh_source(baseline, 8, 99.9) == ""
    assert (
        post_interaction_raw_occupancy_fresh_source(baseline, 8, 100.1)
        == "header_stamp"
    )
    raw_barrier = PostInteractionRawMapBarrier(
        receipt_count=8,
        header_seq=71,
        header_stamp_sec=100.1,
        planning_occupancy_receipt_count=12,
    )

    # The planning grid must arrive after that raw callback and retain its
    # source timestamp (semantic_mapping copies the raw map header).
    assert (
        post_interaction_planning_occupancy_fresh_source(
            raw_barrier, 12, 100.1
        )
        == ""
    )
    assert (
        post_interaction_planning_occupancy_fresh_source(
            raw_barrier, 13, 100.0
        )
        == ""
    )
    assert (
        post_interaction_planning_occupancy_fresh_source(
            raw_barrier, 13, 100.1
        )
        == "source_header_stamp"
    )

    # Only a global publication after the admitted planning map can release
    # make_plan.  An earlier residual delta does not count.
    assert post_interaction_costmap_receipts_fresh_source(18, 41, 18, 41) == ""
    assert (
        post_interaction_costmap_receipts_fresh_source(18, 41, 18, 42)
        == "costmap_update"
    )


def test_post_open_costmap_gate_prefers_exact_event_then_portal_fallback() -> None:
    assert post_interaction_costmap_baseline_keys(
        "object_skill_000047", "portal_47"
    ) == ("event:object_skill_000047", "portal:portal_47")
    assert post_interaction_costmap_baseline_keys("", "portal_47") == (
        "portal:portal_47",
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


def test_post_open_path_waits_for_transient_planner_availability() -> None:
    assert post_open_path_retryable_preflight_reason("empty_plan")
    assert post_open_path_retryable_preflight_reason("endpoint_mismatch")
    assert post_open_path_retryable_preflight_reason("service_unavailable")
    assert post_open_path_retryable_preflight_reason("pose_unavailable")
    assert not post_open_path_retryable_preflight_reason("disabled")


def test_post_open_path_rejects_normal_navigation_fail_open_result() -> None:
    assert post_open_path_is_confirmed(True, "reachable")
    assert not post_open_path_is_confirmed(True, "service_unavailable")
    assert not post_open_path_is_confirmed(True, "pose_unavailable")
    assert not post_open_path_is_confirmed(False, "empty_plan")


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


def test_drawer_interaction_waits_for_post_arrival_public_scan_frame() -> None:
    machine = BehaviorExecutionStateMachine(
        ExecutionConfig(drawer_scan_wait_timeout_s=2.0)
    )
    candidate = interaction_candidate()
    commands = machine.start(candidate, now=0.0)
    assert commands[0]["kind"] == "navigate"

    commands = machine.on_navigation_result(
        True,
        now=1.0,
        wait_for_drawer_scan=True,
    )
    assert machine.state == STATE_WAITING_FOR_DRAWER_SCAN
    assert commands[0]["kind"] == "wait_for_drawer_scan"
    assert machine.timeout_reason(now=2.9) == ""
    assert machine.timeout_reason(now=3.1) == "drawer_scan_fresh_frame_timeout"

    fresh_candidate = {
        **candidate,
        "interaction_command": {
            "sequence_type": "drawer_scan",
            "drawer_container_bbox_2d": [10.0, 20.0, 30.0, 40.0],
            "drawer_container_capture_step": 42,
        },
    }
    commands = machine.on_drawer_scan_ready(fresh_candidate, now=1.2)
    assert machine.state == STATE_INTERACTING
    assert commands[0]["kind"] == "publish_drawer_scan"
    assert commands[0]["candidate"]["interaction_command"][
        "drawer_container_capture_step"
    ] == 42


def test_drawer_interaction_fresh_frame_timeout_is_explicit() -> None:
    machine = BehaviorExecutionStateMachine()
    machine.start(interaction_candidate(), now=0.0)
    machine.on_navigation_result(True, now=1.0, wait_for_drawer_scan=True)

    terminal = machine.on_drawer_scan_wait_failed(
        {"reason": "drawer_scan_fresh_frame_timeout", "last_reason": "rgb_image_not_fresh"},
        now=2.0,
    )

    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is False
    assert machine.error == "drawer_scan_fresh_frame_timeout"


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
