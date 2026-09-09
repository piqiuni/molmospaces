import math

from semantic_decision_py_pkg.step_command_gate import StepCommandGate
from semantic_decision_py_pkg.startup_scan_lifecycle import StartupScanLifecycle
from semantic_decision_py_pkg.startup_scan_timing import (
    startup_scan_elapsed_control_s,
    startup_scan_timeout_reason,
)
from semantic_decision_py_pkg.behavior_execution import (
    BehaviorExecutionStateMachine,
    ExecutionConfig,
    NavigationProgressWatchdog,
    SemanticNavigationProgressSupervisor,
    PostInteractionCostmapBaseline,
    PostInteractionRawMapBarrier,
    STATE_APPROACH_INTERACTION,
    STATE_FINALIZING_EXPLORE,
    STATE_INTERACTING,
    STATE_NAVIGATING,
    STATE_PREPARING_EXPLORE,
    STATE_SCANNING,
    STATE_SUCCEEDED,
    STATE_WAITING_FOR_DRAWER_SCAN,
    STATE_WAITING_FOR_INTERACTION_OBSERVATION,
    STATE_VERIFYING,
    bounded_empty_plan_retry_delay,
    candidate_with_effective_interaction_approach,
    container_two_stage_action_goal_options_for_staging,
    container_two_stage_m1_anchor_priority,
    container_two_stage_face_indices,
    container_two_stage_m1_preflight_batch_indices,
    committed_turn_sign,
    interaction_pose_validation,
    interaction_observation_disposition,
    is_interaction_pose_precondition_failure,
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


def test_container_m1_preflight_batch_scans_remaining_face_diverse_anchors() -> None:
    candidate = {
        "metadata": {
            "container_two_stage_approach": True,
            "container_staging_goal_xyyaw_candidates": [
                [float(index), 0.0, 0.0] for index in range(6)
            ],
            "container_m1_viewpoint_order": [0, 3, 1, 4, 2, 5],
            "container_m1_unavailable_staging_indices": [0],
            "interaction_observation_viewpoint_staging_indices": [3],
        }
    }

    assert container_two_stage_m1_preflight_batch_indices(candidate, 1) == [
        1,
        4,
        2,
        5,
    ]


def test_rejected_container_face_excludes_all_anchors_from_preflight() -> None:
    labels = [
        "aabb_fan_pos_x_angle_+0_clearance_0.50",
        "aabb_fan_pos_x_angle_+15_clearance_0.50",
        "aabb_fan_pos_y_angle_+0_clearance_0.50",
        "aabb_fan_neg_x_angle_+0_clearance_0.50",
    ]
    candidate = {
        "metadata": {
            "container_two_stage_approach": True,
            "container_staging_goal_xyyaw_candidates": [
                [float(index), 0.0, 0.0] for index in range(len(labels))
            ],
            "container_staging_pose_labels": labels,
            "container_m1_viewpoint_order": [0, 2, 3, 1],
            "container_m1_rejected_face_staging_indices": [0, 1],
        }
    }

    assert container_two_stage_face_indices(candidate, 0) == [0, 1]
    assert container_two_stage_m1_preflight_batch_indices(candidate, 0) == [2, 3]


def test_drawer_anchor_priority_prefers_near_then_straight() -> None:
    labels = [
        "aabb_fan_pos_x_angle_-30_clearance_0.60",
        "aabb_fan_pos_x_angle_+15_clearance_0.60",
        "aabb_fan_pos_x_angle_+0_clearance_0.60",
        "aabb_fan_pos_x_angle_-15_clearance_0.95",
    ]
    candidate = {"metadata": {"container_staging_pose_labels": labels}}

    ranked = sorted(
        range(len(labels)),
        key=lambda index: container_two_stage_m1_anchor_priority(candidate, index),
    )

    assert ranked == [2, 1, 0, 3]


def test_pre_m1_empty_anchor_batch_defers_without_candidate_exclusion() -> None:
    machine = BehaviorExecutionStateMachine()
    candidate = {
        "decision_id": "decision-pre-m1-empty",
        "candidate_id": "interaction:drawer:open",
        "behavior_type": "INTERACT",
        "goal_xyyaw": [1.0, 0.0, 0.0],
        "interaction_command": {"action": "open"},
        "metadata": {
            "requires_approach": True,
            "container_two_stage_approach": True,
            "container_two_stage_phase": "staging",
            "m1_observation_staging_required": True,
            "drawer_pre_action_observation": True,
            "interaction_observation_attempts": 0,
        },
    }
    machine.start(candidate, now=0.0)

    commands = machine.defer_container_m1_viewpoint_navigation(
        {"reason": "make_plan_unreachable"}, now=1.0
    )

    assert [command["kind"] for command in commands] == ["terminal"]
    detail = commands[0]["detail"]
    assert detail["m1_evidence_inconclusive"] is True
    assert detail["retryable"] is True
    assert detail["terminal_candidate_exclusion"] is False
    assert detail["m1_capture_not_reached"] is True



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
    bounded = NavigationProgressWatchdog(
        timeout_s=12.0,
        min_displacement_m=0.10,
        allow_yaw_progress=False,
    )
    bounded.reset((0.0, 0.0, 0.0), now=0.0)
    assert not bounded.observe((0.0, 0.0, 0.40), now=11.9)
    # A yaw-only DWA oscillation must eventually cancel/replan instead of
    # resetting the ordinary navigation watchdog.
    assert bounded.observe((0.0, 0.0, -0.40), now=12.0)


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


def test_navigation_progress_watchdog_uses_public_task_steps_when_available() -> None:
    watchdog = NavigationProgressWatchdog(
        timeout_s=1.0,
        timeout_task_steps=6,
        min_displacement_m=0.10,
        allow_yaw_progress=False,
    )
    watchdog.reset((0.0, 0.0, 0.0), now=0.0, task_step_index=10)

    # Host time can advance arbitrarily while VLM/simulator workers contend;
    # only evaluator actions consume the authoritative navigation budget.
    assert not watchdog.observe(
        (0.0, 0.0, 0.0), now=100.0, task_step_index=15
    )
    assert watchdog.observe(
        (0.0, 0.0, 0.0), now=100.1, task_step_index=16
    )

    watchdog.reset((0.0, 0.0, 0.0), now=0.0, task_step_index=20)
    assert not watchdog.observe(
        (0.11, 0.0, 0.0), now=50.0, task_step_index=25
    )
    assert not watchdog.observe(
        (0.11, 0.0, 0.0), now=100.0, task_step_index=30
    )


def test_semantic_progress_supervisor_survives_worker_and_anchor_changes() -> None:
    supervisor = SemanticNavigationProgressSupervisor(
        subgoal_timeout_task_steps=6,
        mission_timeout_task_steps=18,
        min_displacement_m=0.10,
    )

    first = supervisor.observe(
        subgoal_key="drawer|staging|0",
        pose=(0.0, 0.0, 0.0),
        task_step_index=10,
        goal_distance_m=2.0,
    )
    assert not first["subgoal_stalled"]
    # A private waypoint worker retains the same semantic key, so it does not
    # restart the k-step timer.
    stalled = supervisor.observe(
        subgoal_key="drawer|staging|0",
        pose=(0.01, 0.0, 0.0),
        task_step_index=16,
        goal_distance_m=1.995,
    )
    assert stalled["subgoal_stalled"]
    assert not stalled["mission_stalled"]

    # Changing anchor resets only the local timer.  The 3*k mission timer is
    # intentionally continuous while the base remains in the same place.
    assert not supervisor.observe(
        subgoal_key="drawer|staging|15",
        pose=(0.01, 0.0, 0.0),
        task_step_index=17,
        goal_distance_m=1.5,
    )["subgoal_stalled"]
    mission_stall = supervisor.observe(
        subgoal_key="fridge|staging|3",
        pose=(0.02, 0.0, 0.0),
        task_step_index=28,
        goal_distance_m=3.0,
    )
    assert mission_stall["mission_stalled"]


def test_semantic_progress_supervisor_accepts_shortest_yaw_and_translation_progress() -> None:
    supervisor = SemanticNavigationProgressSupervisor(
        subgoal_timeout_task_steps=6,
        mission_timeout_task_steps=18,
        min_displacement_m=0.10,
        min_yaw_error_reduction_rad=0.02,
    )
    supervisor.observe(
        subgoal_key="door|navigation|0",
        pose=(0.0, 0.0, 0.0),
        task_step_index=0,
        goal_distance_m=1.0,
        yaw_error_rad=1.0,
        allow_yaw_progress=True,
    )
    assert not supervisor.observe(
        subgoal_key="door|navigation|0",
        pose=(0.0, 0.0, 0.2),
        task_step_index=5,
        goal_distance_m=1.0,
        yaw_error_rad=0.95,
        allow_yaw_progress=True,
    )["subgoal_stalled"]
    assert not supervisor.observe(
        subgoal_key="door|navigation|0",
        pose=(0.11, 0.0, 0.2),
        task_step_index=11,
        goal_distance_m=0.9,
        yaw_error_rad=0.95,
        allow_yaw_progress=True,
    )["mission_stalled"]


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


def test_interaction_pose_poll_failure_advances_to_next_preserved_option() -> None:
    kwargs = {
        "behavior_type": "INTERACT",
        "failure_detail": {"reason": "interaction_pose_poll_exhausted"},
        "selected_option_index": 2,
        "attempted_navigation_count": 1,
        "max_navigation_attempts": 3,
        "goal_option_count": 5,
    }
    assert next_interaction_approach_option_index(**kwargs) == 3
    assert (
        next_interaction_approach_option_index(
            **{**kwargs, "attempted_navigation_count": 3}
        )
        is None
    )


def test_visual_reposition_advances_to_next_preserved_option() -> None:
    kwargs = {
        "behavior_type": "INTERACT",
        "failure_detail": {"reason": "visual_reposition_required"},
        "selected_option_index": 0,
        "attempted_navigation_count": 1,
        "max_navigation_attempts": 4,
        "goal_option_count": 4,
    }
    assert next_interaction_approach_option_index(**kwargs) == 1
    assert (
        next_interaction_approach_option_index(
            **{**kwargs, "attempted_navigation_count": 4}
        )
        is None
    )


def test_effective_interaction_approach_replaces_bridge_pose_not_primary_goal() -> None:
    primary = [6.8591275, 4.4671125, -math.pi / 2.0]
    fallback = [6.8591275, 4.9671125, -math.pi / 2.0]
    candidate = {
        "candidate_id": "interaction:door_0003:open",
        "behavior_type": "INTERACT",
        "goal_xyyaw": primary,
        "interaction_command": {
            "interaction_approach_pose_xyyaw": primary,
        },
        "metadata": {"goal_xyyaw_candidates": [primary, fallback]},
    }
    bound = candidate_with_effective_interaction_approach(
        candidate,
        fallback,
        goal_option_index=1,
        attempts=[{"index": 1, "goal_xyyaw": fallback}],
    )
    assert bound["goal_xyyaw"] == primary
    assert bound["interaction_command"]["interaction_approach_pose_xyyaw"] == fallback
    assert bound["metadata"]["effective_interaction_approach_pose_xyyaw"] == fallback
    assert bound["metadata"]["interaction_approach_goal_option_index"] == 1

    # This is the failing House7 geometry: the robot reached the fallback,
    # so it must be checked against the fallback rather than the primary.
    actual = [6.6812474, 5.0013154, -1.3780854]
    assert not interaction_pose_validation(
        primary,
        actual,
        distance_tolerance_m=0.45,
        yaw_tolerance_rad=0.55,
    )["valid"]
    assert interaction_pose_validation(
        fallback,
        actual,
        distance_tolerance_m=0.45,
        yaw_tolerance_rad=0.55,
    )["valid"]


def test_pose_precondition_failure_is_not_an_object_failure() -> None:
    assert is_interaction_pose_precondition_failure(
        {"failure_reason": "interaction_pose_invalid"}
    )
    assert is_interaction_pose_precondition_failure(
        {"reason": "interaction_pose_poll_exhausted"}
    )
    assert not is_interaction_pose_precondition_failure(
        {"failure_reason": "articulation_resolution_failed"}
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


def observation_required_interaction_candidate(requires_approach=True):
    candidate = interaction_candidate(requires_approach=requires_approach)
    candidate["candidate_id"] = "interaction:portal_1:open"
    candidate["target_id"] = "portal_1"
    candidate["target_name"] = "door_0001"
    candidate["interaction_command"].update(
        {"node_id": "portal_1", "object_id": "door_0001", "action": "open"}
    )
    candidate["metadata"].update(
        {
            "node_type": "portal",
            "observation_required": True,
            "reobserve": True,
            "observation_reason": "mllm_portal_state_unknown",
            "interaction_observation_max_attempts": 2,
            "interaction_observation_source": "mllm_attribute_inference",
        }
    )
    return candidate


def drawer_pre_action_candidate(requires_approach=False):
    candidate = interaction_candidate(requires_approach=requires_approach)
    candidate["candidate_id"] = "interaction:drawer_1:open"
    candidate["target_id"] = "drawer_1"
    candidate["target_name"] = "chestofdrawers_asset"
    candidate["interaction_command"].update(
        {
            "node_id": "drawer_1",
            "object_id": "drawer_1",
            "action": "scan",
            "sequence_type": "drawer_scan",
        }
    )
    candidate["metadata"].update(
        {
            "node_type": "container",
            "observation_required": True,
            "reobserve": True,
            "drawer_pre_action_observation": True,
            "interaction_observation_max_attempts": 2,
            "interaction_observation_source": "mllm_attribute_inference",
        }
    )
    if requires_approach:
        candidate["metadata"]["goal_xyyaw_candidates"] = [
            [1.0, 2.0, 0.0],
            [2.0, 2.0, 1.57],
            [2.0, 3.0, 3.14],
        ]
    return candidate


def container_pre_action_candidate(requires_approach=True):
    candidate = interaction_candidate(requires_approach=requires_approach)
    candidate["candidate_id"] = "interaction:fridge_1:open"
    candidate["target_id"] = "fridge_1"
    candidate["target_name"] = "refrigerator_asset"
    candidate["interaction_command"].update(
        {"node_id": "fridge_1", "object_id": "fridge_1", "action": "open"}
    )
    candidate["metadata"].update(
        {
            "node_type": "container",
            "observation_required": True,
            "reobserve": True,
            "container_pre_action_observation": True,
            "interaction_observation_max_attempts": 4,
            "interaction_observation_source": "mllm_attribute_inference",
            "goal_xyyaw_candidates": [
                [1.0, 2.0, 0.0],
                [2.0, 2.0, 1.57],
                [2.0, 3.0, 3.14],
                [1.0, 3.0, -1.57],
            ],
        }
    )
    return candidate


def two_stage_container_pre_action_candidate():
    candidate = container_pre_action_candidate()
    candidate["interaction_command"].update(
        {
            "container_staging_ready_distance_m": 0.30,
            "container_physical_action_ready_distance_m": 0.18,
            "interaction_ready_distance_m": 0.30,
        }
    )
    staging_goals = [
        [1.0, 2.0, 0.0],
        [2.0, 2.0, 1.57],
        [2.0, 3.0, 3.14],
        [1.0, 3.0, -1.57],
    ]
    action_goals = [
        [1.30, 2.0, 0.0],
        [2.30, 2.0, 1.57],
        [2.30, 3.0, 3.14],
        [1.30, 3.0, -1.57],
    ]
    candidate["metadata"].update(
        {
            "m1_observation_staging_required": True,
            "container_two_stage_approach": True,
            "container_two_stage_phase": "staging",
            "container_staging_goal_xyyaw_candidates": staging_goals,
            "container_staging_pose_labels": [
                "safe_outer",
                "safe_far",
                "safe_farthest",
                "safe_max",
            ],
            "container_action_goal_xyyaw_by_staging_index": action_goals,
            "container_action_pose_labels_by_staging_index": [
                "safe_outer_physical_action",
                "safe_far_physical_action",
                "safe_farthest_physical_action",
                "safe_max_physical_action",
            ],
            "container_action_goal_xyyaw_options_by_staging_index": [
                [
                    action_goals[index],
                    [action_goals[index][0], action_goals[index][1] + 0.22, action_goals[index][2]],
                    [action_goals[index][0], action_goals[index][1] - 0.22, action_goals[index][2]],
                ]
                for index in range(len(action_goals))
            ],
            "container_action_pose_option_labels_by_staging_index": [
                [
                    f"safe_{label}_physical_action" if not label.endswith("physical_action") else label,
                    f"{label}_tangent_left",
                    f"{label}_tangent_right",
                ]
                for label in ["outer", "far", "farthest", "max"]
            ],
            "container_two_stage_staging_observation_required": True,
            "container_two_stage_staging_container_pre_action_observation": True,
            "container_two_stage_staging_drawer_pre_action_observation": False,
            # This is the executor's public, request-ID-matched evidence token;
            # the state machine must not infer it from a ready-looking payload.
            "accepted_container_m1_evidence": {
                "staging_pose_xyyaw": list(staging_goals[0]),
                "capture_pose_xyyaw": list(staging_goals[0]),
                "capture_step": 11,
            },
        }
    )
    return candidate


def direct_capture_two_stage_container_candidate():
    """Return an anchor -> capture -> action candidate for M1 contract tests."""

    candidate = two_stage_container_pre_action_candidate()
    staging = candidate["metadata"]["container_staging_goal_xyyaw_candidates"]
    captures = [
        [1.10, 2.0, 0.0],
        [2.10, 2.0, 1.57],
        [2.10, 3.0, 3.14],
        [1.10, 3.0, -1.57],
    ]
    candidate["metadata"].update(
        {
            "container_two_stage_mapping_ready": True,
            "container_m1_capture_goal_xyyaw_by_staging_index": captures,
            "container_m1_capture_pose_labels_by_staging_index": [
                f"capture_{index}" for index in range(len(captures))
            ],
            "interaction_observation_same_pose_samples_per_view": 2,
            "interaction_observation_max_viewpoints": 2,
            "interaction_observation_max_total_requests": 4,
        }
    )
    candidate["metadata"].pop("accepted_container_m1_evidence", None)
    # The primary public goal remains the navigation anchor, not the M1 pose.
    candidate["goal_xyyaw"] = list(staging[0])
    return candidate


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


def test_interaction_final_align_budget_covers_a_near_door_turn_without_rear_cap() -> None:
    """A slow final-align controller needs more windows than rear pre-turning."""

    # Live failure shape: 2.237 rad initial error, .30 rad/s, fixed .20 s
    # actions, and .15 rad terminal tolerance => ceil(2.087 / .06) = 35.
    assert prerotation_control_step_budget(
        2.237,
        0.15,
        speed_rad_s=0.30,
        control_dt_s=0.20,
        max_control_steps=56,
    ) == 35
    # The dedicated hard cap accommodates a worst-case pi turn with margin;
    # it remains finite and is not the rear-prerotation 12-step cap.
    assert prerotation_control_step_budget(
        math.pi,
        0.15,
        speed_rad_s=0.30,
        control_dt_s=0.20,
        max_control_steps=56,
    ) == 50
    assert prerotation_control_step_budget(
        math.pi,
        0.15,
        speed_rad_s=0.30,
        control_dt_s=0.20,
        max_control_steps=12,
    ) == 12


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


def test_explore_navigation_uses_bounded_prerotation_without_final_yaw_alignment() -> None:
    assert navigation_should_prerotate("EXPLORE")
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
    verification = machine.on_interaction_result(True, now=2.0)
    assert machine.state == STATE_VERIFYING
    assert verification[0]["kind"] == "verify_interaction"
    assert verification[0]["backend_success"] is True
    assert machine.on_graph_state("closed", now=3.0) == []
    terminal = machine.on_graph_state("open", now=4.0)
    assert machine.state == STATE_SUCCEEDED
    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is True


def test_backend_success_finishes_without_waiting_for_visual_audit() -> None:
    machine = BehaviorExecutionStateMachine()
    machine.start(interaction_candidate(requires_approach=False), now=0.0)
    assert machine.state == STATE_INTERACTING
    machine.on_interaction_result(True, {"post_state": "open"}, now=1.0)
    assert machine.state == STATE_VERIFYING

    terminal = machine.on_backend_result(
        True,
        {"post_state": "open", "event_id": "interaction_001"},
        now=1.1,
    )
    assert machine.state == STATE_SUCCEEDED
    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is True
    assert terminal[0]["detail"]["visual_audit_pending"] is True


def test_unknown_portal_reobserves_after_approach_before_physical_action() -> None:
    machine = BehaviorExecutionStateMachine()
    commands = machine.start(observation_required_interaction_candidate(), now=0.0)
    assert commands[0]["kind"] == "navigate"

    commands = machine.on_navigation_result(
        True, {"capture_step": 10}, now=1.0
    )
    assert machine.state == STATE_WAITING_FOR_INTERACTION_OBSERVATION
    assert commands[0]["kind"] == "request_interaction_observation"
    assert commands[0]["attempt"] == 1
    assert commands[0]["min_capture_step"] == 11
    assert commands[0]["require_current_visibility"] is True
    assert commands[0]["required_attribute_source"] == "mllm_attribute_inference"

    commands = machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "state": "closed",
            "attribute_capture_step": 11,
        },
        now=2.0,
    )
    assert machine.state == STATE_INTERACTING
    assert commands[0]["kind"] == "interact"
    assert commands[0]["observation"]["state"] == "closed"
    assert machine.candidate["metadata"]["observation_required"] is False


def test_unknown_portal_open_observation_finishes_without_action() -> None:
    machine = BehaviorExecutionStateMachine()
    commands = machine.start(
        observation_required_interaction_candidate(requires_approach=False), now=0.0
    )
    assert commands[0]["kind"] == "request_interaction_observation"

    terminal = machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "state": "static_open",
            "attribute_capture_step": 1,
        },
        now=1.0,
    )
    assert machine.state == STATE_SUCCEEDED
    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is True
    assert terminal[0]["detail"]["action_executed"] is False
    assert terminal[0]["detail"]["observation_outcome"] == "finish_without_action"


def test_unknown_portal_observation_retries_then_terminates_unresolved() -> None:
    machine = BehaviorExecutionStateMachine()
    machine.start(observation_required_interaction_candidate(requires_approach=False), now=0.0)

    commands = machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "state": "unknown",
            "attribute_capture_step": 1,
        },
        now=1.0,
    )
    assert machine.state == STATE_WAITING_FOR_INTERACTION_OBSERVATION
    assert commands[0]["kind"] == "request_interaction_observation"
    assert commands[0]["attempt"] == 2

    terminal = machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "state": "unknown",
            "attribute_capture_step": 2,
        },
        now=2.0,
    )
    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is False
    assert terminal[0]["detail"]["reason"] == "interaction_observation_unresolved"


def test_container_pre_action_confirms_one_negative_before_moving_to_next_view() -> None:
    machine = BehaviorExecutionStateMachine()
    commands = machine.start(container_pre_action_candidate(), now=0.0)
    assert commands[0]["kind"] == "navigate"
    request = machine.on_navigation_result(True, {"capture_step": 10}, now=0.5)
    assert request[0]["kind"] == "request_interaction_observation"
    assert request[0]["min_capture_step"] == 11

    same_pose = machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "state": "closed",
            "view_state": "side_or_back",
            "front_surface_visible": False,
            "approach_ready": False,
            "attribute_capture_step": 11,
            "container_visual_precondition_reason": "m1_view_state_side_or_back",
        },
        now=1.0,
    )
    assert machine.state == STATE_WAITING_FOR_INTERACTION_OBSERVATION
    assert same_pose[0]["kind"] == "request_interaction_observation"
    assert same_pose[0]["same_pose_sample"] == 2

    retry = machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "state": "closed",
            "view_state": "side_or_back",
            "front_surface_visible": False,
            "approach_ready": False,
            "attribute_capture_step": 12,
            "container_visual_precondition_reason": "m1_view_state_side_or_back",
        },
        now=1.1,
    )
    assert machine.state == STATE_APPROACH_INTERACTION
    assert retry[0]["kind"] == "navigate"
    assert retry[0]["start_goal_option_index"] == 1
    assert retry[0]["reason"] == "container_visual_reobserve_next_approach"

    second_request = machine.on_navigation_result(
        True, {"capture_step": 20}, now=1.5
    )
    assert second_request[0]["kind"] == "request_interaction_observation"
    execute = machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "state": "closed",
            "view_state": "front",
            "front_surface_visible": True,
            "approach_ready": True,
            "observed_bbox_2d": [10, 10, 80, 120],
            "attribute_capture_step": 21,
            "container_visual_precondition_reason": "ready",
        },
        now=2.0,
    )
    assert machine.state == STATE_INTERACTING
    assert execute[0]["kind"] == "interact"


def test_container_two_stage_m1_staging_navigates_inner_before_bridge() -> None:
    machine = BehaviorExecutionStateMachine()
    candidate = two_stage_container_pre_action_candidate()
    commands = machine.start(candidate, now=0.0)
    assert commands[0]["kind"] == "navigate"

    first_request = machine.on_navigation_result(
        True, {"capture_step": 10}, now=0.5
    )
    assert first_request[0]["kind"] == "request_interaction_observation"

    inner_navigation = machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "state": "closed",
            "view_state": "front",
            "front_surface_visible": True,
            "approach_ready": True,
            "observed_bbox_2d": [10, 10, 80, 120],
            "attribute_capture_step": 11,
            "container_visual_precondition_reason": "ready",
        },
        now=1.0,
    )

    assert machine.state == STATE_APPROACH_INTERACTION
    assert [command["kind"] for command in inner_navigation] == ["navigate"]
    assert inner_navigation[0]["reason"] == (
        "container_m1_ready_navigate_physical_action_pose"
    )
    assert machine.candidate["goal_xyyaw"] == [1.30, 2.0, 0.0]
    assert machine.candidate["interaction_command"][
        "interaction_approach_pose_xyyaw"
    ] == [1.30, 2.0, 0.0]
    assert machine.candidate["interaction_command"]["interaction_ready_distance_m"] == 0.18
    assert machine.candidate["metadata"]["container_two_stage_phase"] == "physical_action"
    assert machine.candidate["metadata"]["observation_required"] is False
    assert machine.candidate["metadata"]["container_pre_action_observation"] is False

    # The inner arrival emits the physical command directly.  A second M1
    # request here would re-observe from the close action pose and violate the
    # outer-evidence contract.
    bridge = machine.on_navigation_result(True, {"capture_step": 12}, now=1.5)
    assert machine.state == STATE_INTERACTING
    assert [command["kind"] for command in bridge] == ["interact"]


def test_shared_container_anchor_executes_without_second_navigation() -> None:
    machine = BehaviorExecutionStateMachine()
    candidate = two_stage_container_pre_action_candidate()
    anchors = candidate["metadata"]["container_staging_goal_xyyaw_candidates"]
    candidate["metadata"].update(
        {
            "container_anchor_shared_pose": True,
            "container_m1_front_axis_from_capture": False,
            "container_m1_capture_goal_xyyaw_by_staging_index": [
                list(goal) for goal in anchors
            ],
            "container_action_goal_xyyaw_by_staging_index": [
                list(goal) for goal in anchors
            ],
            "container_action_goal_xyyaw_options_by_staging_index": [
                [list(goal)] for goal in anchors
            ],
        }
    )
    candidate["interaction_command"].update(
        {
            "container_staging_ready_distance_m": 0.10,
            "container_physical_action_ready_distance_m": 0.10,
            "interaction_ready_distance_m": 0.10,
        }
    )
    machine.start(candidate, now=0.0)
    request = machine.on_navigation_result(True, {"capture_step": 10}, now=0.5)
    assert request[0]["kind"] == "request_interaction_observation"

    command = machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "state": "closed",
            "view_state": "front",
            "front_surface_visible": True,
            "approach_ready": True,
            "observed_bbox_2d": [10, 10, 80, 120],
            "attribute_capture_step": 11,
            "container_visual_precondition_reason": "ready",
        },
        now=1.0,
    )

    assert machine.state == STATE_INTERACTING
    assert [item["kind"] for item in command] == ["interact"]
    assert command[0]["reason"] == "container_m1_ready_at_shared_action_anchor"
    assert machine.candidate["goal_xyyaw"] == anchors[0]
    assert machine.candidate["metadata"]["container_two_stage_action_geometry_source"] == (
        "shared_selected_anchor"
    )


def test_container_m1_repeats_fresh_same_capture_then_moves_to_next_anchor() -> None:
    """One rejected M1 image is not an exclusion or an inner-action shortcut."""

    machine = BehaviorExecutionStateMachine()
    candidate = direct_capture_two_stage_container_candidate()
    anchor0 = candidate["metadata"]["container_staging_goal_xyyaw_candidates"][0]
    capture0 = candidate["metadata"]["container_m1_capture_goal_xyyaw_by_staging_index"][0]
    capture1 = candidate["metadata"]["container_m1_capture_goal_xyyaw_by_staging_index"][1]
    commands = machine.start(candidate, now=0.0)
    assert commands[0]["kind"] == "navigate"
    assert commands[0]["candidate"]["goal_xyyaw"] == anchor0

    # Anchor arrival cannot issue M1; it first moves to the direct capture pose.
    capture_navigation = machine.on_navigation_result(True, {}, now=0.1)
    assert capture_navigation[0]["kind"] == "navigate"
    assert capture_navigation[0]["reason"] == "container_navigation_anchor_to_m1_capture"
    assert capture_navigation[0]["candidate"]["goal_xyyaw"] == capture0
    assert machine.candidate["metadata"]["container_two_stage_phase"] == "m1_capture"
    # The direct visual pose uses the physical/capture tolerance, not the
    # wider navigation-anchor tolerance retained for the outer recovery point.
    assert machine.candidate["interaction_command"]["interaction_ready_distance_m"] == (
        machine.candidate["interaction_command"]["container_physical_action_ready_distance_m"]
    )

    request1 = machine.on_navigation_result(True, {}, now=0.2)
    assert request1[0]["kind"] == "request_interaction_observation"
    assert request1[0]["same_pose_sample"] == 1
    assert request1[0]["viewpoint"] == 1

    rejected = {
        "attribute_source": "mllm_attribute_inference",
        "attribute_status": "ready",
        "is_currently_visible": True,
        "state": "closed",
        "view_state": "oblique",
        "front_surface_visible": False,
        "approach_ready": False,
        "capture_step": 12,
    }
    request2 = machine.on_interaction_observation_result(rejected, now=0.3)
    assert request2[0]["kind"] == "request_interaction_observation"
    assert request2[0]["same_pose_sample"] == 2
    assert request2[0]["viewpoint"] == 1
    assert machine.candidate["goal_xyyaw"] == capture0

    # Only the second negative consumes a new viewpoint.  It returns to the
    # next navigation anchor, never calls M1 from that offset, and never emits
    # a terminal candidate exclusion.
    next_anchor = machine.on_interaction_observation_result(
        {**rejected, "capture_step": 13}, now=0.4
    )
    assert next_anchor[0]["kind"] == "navigate"
    assert next_anchor[0]["start_goal_option_index"] == 1
    assert next_anchor[0]["reason"] == "container_m1_capture_failed_next_outer_staging"
    assert machine.candidate["metadata"]["container_two_stage_phase"] == "staging"

    capture_navigation_1 = machine.on_navigation_result(True, {}, now=0.5)
    assert capture_navigation_1[0]["candidate"]["goal_xyyaw"] == capture1
    request3 = machine.on_navigation_result(True, {}, now=0.6)
    assert request3[0]["kind"] == "request_interaction_observation"
    assert request3[0]["viewpoint"] == 2
    assert request3[0]["same_pose_sample"] == 1

    # The executor normally supplies this request-ID/pose-bound record.  The
    # state machine must use the direct capture pose, not the outer anchor.
    machine.candidate["metadata"]["accepted_container_m1_evidence"] = {
        "staging_pose_xyyaw": list(capture1),
        "capture_pose_xyyaw": list(capture1),
        "capture_step": 14,
    }
    action_navigation = machine.on_interaction_observation_result(
        {
            **rejected,
            "capture_step": 14,
            "view_state": "front",
            "front_surface_visible": True,
            "approach_ready": True,
        },
        now=0.7,
    )
    assert action_navigation[0]["kind"] == "navigate"
    assert action_navigation[0]["reason"] == "container_m1_ready_navigate_physical_action_pose"
    assert machine.candidate["metadata"]["container_two_stage_phase"] == "physical_action"
    assert machine.candidate["metadata"]["container_m1_evidence_capture_pose_xyyaw"] == capture1


def test_distinct_m1_capture_requires_real_arrival_without_spending_view_budget() -> None:
    """A failed direct capture cannot masquerade as a second sampled camera view."""

    machine = BehaviorExecutionStateMachine(
        ExecutionConfig(
            container_m1_distinct_view_arrival_tolerance_m=0.05,
            container_m1_distinct_view_arrival_yaw_tolerance_rad=0.08,
        )
    )
    candidate = direct_capture_two_stage_container_candidate()
    capture0 = candidate["metadata"]["container_m1_capture_goal_xyyaw_by_staging_index"][0]
    capture1 = candidate["metadata"]["container_m1_capture_goal_xyyaw_by_staging_index"][1]
    machine.start(candidate, now=0.0)

    # Anchor 0 -> direct capture 0 -> two fresh negative M1 samples.
    machine.on_navigation_result(True, {}, now=0.1)
    request1 = machine.on_navigation_result(True, {}, now=0.2)
    assert request1[0]["kind"] == "request_interaction_observation"
    assert machine.candidate["metadata"]["container_m1_last_sampled_capture_goal_xyyaw"] == capture0
    assert machine.candidate["metadata"]["container_m1_capture_requires_distinct_view_arrival"] is False
    rejected = {
        "attribute_source": "mllm_attribute_inference",
        "attribute_status": "ready",
        "is_currently_visible": True,
        "state": "closed",
        "view_state": "oblique",
        "front_surface_visible": False,
        "approach_ready": False,
        "capture_step": 12,
    }
    machine.on_interaction_observation_result(rejected, now=0.3)
    next_anchor = machine.on_interaction_observation_result(
        {**rejected, "capture_step": 13}, now=0.4
    )
    assert next_anchor[0]["start_goal_option_index"] == 1

    # Anchor 1 is not itself an M1 viewpoint. Its paired capture is different
    # from the pose which actually issued the first M1 request, so navigation
    # must prove stricter direct-capture arrival before M1 can be requested.
    capture_navigation = machine.on_navigation_result(True, {}, now=0.5)
    assert capture_navigation[0]["candidate"]["goal_xyyaw"] == capture1
    metadata = machine.candidate["metadata"]
    assert metadata["container_m1_capture_requires_distinct_view_arrival"] is True
    assert metadata["container_m1_capture_prior_sampled_goal_xyyaw"] == capture0
    assert metadata["interaction_observation_attempts"] == 2
    assert metadata["interaction_observation_viewpoint_count"] == 1
    assert metadata["interaction_observation_viewpoint_staging_indices"] == [0]

    # If direct capture 1 cannot reach its exact pose, no third M1 request is
    # issued and therefore neither samples nor viewpoints are consumed.
    retry = machine.retry_container_two_stage_staging(
        next_staging_goal_option_index=2,
        interaction_approach_attempts=[{"index": 1, "phase": "m1_capture"}],
        detail={"reason": "interaction_pose_poll_exhausted"},
        now=0.6,
    )
    assert retry[0]["kind"] == "navigate"
    metadata = machine.candidate["metadata"]
    assert metadata["container_two_stage_phase"] == "staging"
    assert metadata["interaction_observation_attempts"] == 2
    assert metadata["interaction_observation_viewpoint_count"] == 1
    assert metadata["interaction_observation_viewpoint_staging_indices"] == [0]


def test_container_m1_budget_defers_without_terminal_candidate_exclusion() -> None:
    machine = BehaviorExecutionStateMachine()
    candidate = direct_capture_two_stage_container_candidate()
    candidate["metadata"].update(
        {
            "interaction_observation_max_viewpoints": 1,
            "interaction_observation_max_total_requests": 2,
        }
    )
    machine.start(candidate, now=0.0)
    machine.on_navigation_result(True, {}, now=0.1)
    machine.on_navigation_result(True, {}, now=0.2)
    rejected = {
        "attribute_source": "mllm_attribute_inference",
        "attribute_status": "ready",
        "is_currently_visible": True,
        "view_state": "oblique",
        "front_surface_visible": False,
        "approach_ready": False,
        "capture_step": 12,
    }
    same_pose = machine.on_interaction_observation_result(rejected, now=0.3)
    assert same_pose[0]["kind"] == "request_interaction_observation"
    terminal = machine.on_interaction_observation_result(
        {**rejected, "capture_step": 13}, now=0.4
    )
    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is False
    assert terminal[0]["detail"]["m1_evidence_inconclusive"] is True
    assert terminal[0]["detail"]["retryable"] is True
    assert terminal[0]["detail"]["terminal_candidate_exclusion"] is False


def test_container_two_stage_accepted_m1_uses_same_face_inner_options_before_outer() -> None:
    machine = BehaviorExecutionStateMachine()
    candidate = two_stage_container_pre_action_candidate()
    machine.start(candidate, now=0.0)
    machine.on_navigation_result(True, {"capture_step": 10}, now=0.5)
    inner_navigation = machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "state": "closed",
            "view_state": "front",
            "front_surface_visible": True,
            "approach_ready": True,
            "observed_bbox_2d": [10, 10, 80, 120],
            "attribute_capture_step": 11,
            "container_visual_precondition_reason": "ready",
        },
        now=1.0,
    )

    assert inner_navigation[0]["start_goal_option_index"] == 0
    assert navigation_goal_options(machine.candidate) == [
        (1.30, 2.0, 0.0),
        (1.30, 2.22, 0.0),
        (1.30, 1.78, 0.0),
    ]
    assert machine.candidate["metadata"]["container_two_stage_action_goal_option_index"] == 0
    assert container_two_stage_action_goal_options_for_staging(machine.candidate, 0) == [
        (1.30, 2.0, 0.0),
        (1.30, 2.22, 0.0),
        (1.30, 1.78, 0.0),
    ]


def test_container_two_stage_m1_front_capture_selects_physical_face() -> None:
    """A ready M1 image freezes its actual capture ray as the action face."""

    machine = BehaviorExecutionStateMachine()
    candidate = two_stage_container_pre_action_candidate()
    base_staging = [1.0, 2.0, 0.0]
    tangent_staging = [1.0, 2.2, -0.20]
    base_action = [1.30, 2.0, 0.0]
    base_options = [
        base_action,
        [1.30, 2.22, 0.0],
        [1.30, 1.78, 0.0],
    ]
    candidate["metadata"].update(
        {
            "container_staging_goal_xyyaw_candidates": [
                base_staging,
                tangent_staging,
            ],
            "container_staging_pose_labels": [
                "safe_outer",
                "safe_outer_tangent_left",
            ],
            "container_staging_source_index_by_index": [0, 0],
            "container_action_goal_xyyaw_by_staging_index": [
                base_action,
                list(base_action),
            ],
            "container_action_pose_labels_by_staging_index": [
                "safe_outer_physical_action",
                "safe_outer_physical_action",
            ],
            "container_action_goal_xyyaw_options_by_staging_index": [
                base_options,
                [list(option) for option in base_options],
            ],
            "container_action_pose_option_labels_by_staging_index": [
                [
                    "safe_outer_physical_action",
                    "safe_outer_physical_action_tangent_left",
                    "safe_outer_physical_action_tangent_right",
                ],
                [
                    "safe_outer_physical_action",
                    "safe_outer_physical_action_tangent_left",
                    "safe_outer_physical_action_tangent_right",
                ],
            ],
            "container_m1_front_axis_from_capture": True,
            "container_geometry_anchor_xy": [0.0, 0.0],
            "container_geometry_aabb_size_xy": [2.0, 2.0],
            "container_physical_action_standoff_m": 0.30,
            "container_action_lateral_offset_m": 0.22,
            "accepted_container_m1_evidence": {
                "staging_pose_xyyaw": list(tangent_staging),
                "capture_pose_xyyaw": list(tangent_staging),
                "capture_step": 11,
                "m1_front_axis_xy": [0.0, 1.0],
                "m1_front_yaw": -math.pi / 2.0,
                "m1_front_axis_source": "m1_confirmed_capture_pose",
            },
        }
    )
    candidate["goal_xyyaw"] = list(base_staging)
    machine.start(candidate, now=0.0)
    request = machine.on_navigation_result(True, {"capture_step": 10}, now=0.5)
    assert request[0]["kind"] == "request_interaction_observation"
    # The executor selects the tangent index after its ordinary make-plan
    # preflight; the state machine must bind the fresh evidence to that exact
    # outer pose before it can select the copied base action mapping.
    machine.candidate["metadata"]["interaction_approach_goal_option_index"] = 1
    machine.candidate["interaction_command"]["interaction_approach_pose_xyyaw"] = list(
        tangent_staging
    )
    commands = machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "state": "closed",
            "view_state": "front",
            "front_surface_visible": True,
            "approach_ready": True,
            "observed_bbox_2d": [10, 10, 80, 120],
            "attribute_capture_step": 11,
            "container_visual_precondition_reason": "ready",
        },
        now=1.0,
    )

    assert machine.state == STATE_APPROACH_INTERACTION
    assert [command["kind"] for command in commands] == ["navigate"]
    assert commands[0]["reason"] == "container_m1_ready_navigate_physical_action_pose"
    assert machine.candidate["metadata"]["container_two_stage_staging_goal_option_index"] == 1
    assert machine.candidate["metadata"]["container_m1_evidence_staging_pose_xyyaw"] == tangent_staging
    # The old index mapping pointed at base_action.  The accepted M1 front
    # image instead makes the calibrated +Y capture ray authoritative for this
    # active decision: surface boundary 1.0 + physical clearance .30.
    assert machine.candidate["goal_xyyaw"] == [0.0, 1.3, -math.pi / 2.0]
    assert navigation_goal_options(machine.candidate) == [
        (0.0, 1.3, -math.pi / 2.0),
        (-0.22, 1.3, math.atan2(-1.3, 0.22)),
        (0.22, 1.3, math.atan2(-1.3, -0.22)),
    ]
    assert (
        machine.candidate["metadata"]["container_two_stage_action_geometry_source"]
        == "m1_confirmed_capture_front_axis"
    )
    assert machine.candidate["interaction_command"]["interaction_approach_axis_xy"] == [
        0.0,
        1.0,
    ]
    assert machine.candidate["metadata"]["m1_observation_staging_required"] is False
    assert machine.candidate["metadata"]["observation_required"] is False


def test_container_two_stage_inner_failure_returns_next_outer_m1_staging() -> None:
    machine = BehaviorExecutionStateMachine()
    candidate = two_stage_container_pre_action_candidate()
    # Model the live failure shape: the last rejected M1 response set an older
    # baseline, then a later front frame authorized the inner physical pose.
    # Returning to another outer stance must never reuse a frame between them.
    candidate["metadata"]["interaction_observation_after_capture_step"] = 207
    candidate["metadata"]["accepted_container_m1_evidence"]["capture_step"] = 306
    machine.start(candidate, now=0.0)
    machine.on_navigation_result(True, {"capture_step": 207}, now=0.5)
    machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "state": "closed",
            "view_state": "front",
            "front_surface_visible": True,
            "approach_ready": True,
            "observed_bbox_2d": [10, 10, 80, 120],
            "attribute_capture_step": 306,
            "container_visual_precondition_reason": "ready",
        },
        now=1.0,
    )

    retry = machine.retry_container_two_stage_staging(
        next_staging_goal_option_index=1,
        interaction_approach_attempts=[
            {"index": 0, "phase": "staging"},
            {"index": 0, "phase": "physical_action"},
        ],
        detail={"reason": "unsafe_open_sweep"},
        now=1.5,
    )

    assert machine.state == STATE_APPROACH_INTERACTION
    assert [command["kind"] for command in retry] == ["navigate"]
    assert retry[0]["start_goal_option_index"] == 1
    assert retry[0]["reason"] == "container_inner_action_failed_next_outer_staging"
    assert machine.candidate["metadata"]["container_two_stage_phase"] == "staging"
    assert machine.candidate["metadata"]["observation_required"] is True
    assert machine.candidate["metadata"]["container_pre_action_observation"] is True
    assert "accepted_container_m1_evidence" not in machine.candidate["metadata"]
    assert (
        machine.candidate["metadata"]["interaction_observation_after_capture_step"]
        == 306
    )
    assert machine.candidate["interaction_command"]["interaction_ready_distance_m"] == 0.30

    # Only after reaching the next outer stance can M1 be requested again.
    next_request = machine.on_navigation_result(
        True, {"interaction_arrival_step_index": 411}, now=2.0
    )
    assert machine.state == STATE_WAITING_FOR_INTERACTION_OBSERVATION
    assert [command["kind"] for command in next_request] == [
        "request_interaction_observation"
    ]
    assert next_request[0]["min_capture_step"] == 307


def test_container_two_stage_missing_inner_mapping_fails_closed_to_next_outer() -> None:
    machine = BehaviorExecutionStateMachine()
    candidate = two_stage_container_pre_action_candidate()
    candidate["metadata"]["container_two_stage_mapping_ready"] = False
    candidate["metadata"]["container_action_goal_xyyaw_by_staging_index"] = []
    # This test isolates the mapping fail-closed branch rather than the normal
    # default same-pose evidence confirmation.
    candidate["metadata"]["interaction_observation_same_pose_samples_per_view"] = 1
    machine.start(candidate, now=0.0)
    machine.on_navigation_result(True, {"capture_step": 10}, now=0.5)

    retry = machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "state": "closed",
            "view_state": "front",
            "front_surface_visible": True,
            "approach_ready": True,
            "observed_bbox_2d": [10, 10, 80, 120],
            "attribute_capture_step": 11,
            "container_visual_precondition_reason": "ready",
        },
        now=1.0,
    )

    assert machine.state == STATE_APPROACH_INTERACTION
    assert [command["kind"] for command in retry] == ["navigate"]
    assert retry[0]["start_goal_option_index"] == 1
    assert all(command["kind"] != "interact" for command in retry)


def test_container_visual_open_without_fresh_bbox_is_reobserved() -> None:
    machine = BehaviorExecutionStateMachine()
    candidate = container_pre_action_candidate()
    candidate["metadata"]["interaction_observation_same_pose_samples_per_view"] = 1
    machine.start(candidate, now=0.0)
    machine.on_navigation_result(True, {"capture_step": 10}, now=0.5)
    retry = machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "state": "open",
            "view_state": "front",
            "front_surface_visible": True,
            "approach_ready": True,
            "attribute_capture_step": 11,
            "container_visual_precondition_reason": "m1_public_bbox_unavailable",
        },
        now=1.0,
    )
    assert machine.state == STATE_APPROACH_INTERACTION
    assert retry[0]["kind"] == "navigate"
    assert retry[0]["start_goal_option_index"] == 1


def test_drawer_pre_action_reobserves_then_defers_inconclusive_m1_evidence() -> None:
    machine = BehaviorExecutionStateMachine()
    commands = machine.start(drawer_pre_action_candidate(requires_approach=True), now=0.0)
    assert commands[0]["kind"] == "navigate"
    initial_observation = machine.on_navigation_result(
        True, {"capture_step": 10}, now=0.5
    )
    assert initial_observation[0]["kind"] == "request_interaction_observation"
    assert initial_observation[0]["attempt"] == 1
    assert initial_observation[0]["min_capture_step"] == 11

    retry = machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "view_state": "side",
            "front_surface_visible": False,
            "approach_ready": False,
            "drawer_action_regions_ready": False,
            "attribute_capture_step": 11,
        },
        now=1.0,
    )
    assert machine.state == STATE_WAITING_FOR_INTERACTION_OBSERVATION
    assert retry[0]["kind"] == "request_interaction_observation"
    assert retry[0]["attempt"] == 2
    assert retry[0]["same_pose_sample"] == 2

    # A second independent negative at the held capture pose exhausts the
    # explicit two-request budget.  It must defer, not permanently exclude the
    # drawer or manufacture a different physical action.
    terminal = machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "view_state": "front",
            "front_surface_visible": True,
            "approach_ready": True,
            "drawer_action_regions_ready": False,
            "attribute_capture_step": 12,
        },
        now=1.5,
    )
    assert machine.state != STATE_VERIFYING
    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is False
    assert terminal[0]["detail"]["m1_evidence_inconclusive"] is True
    assert terminal[0]["detail"]["retryable"] is True
    assert terminal[0]["detail"]["terminal_candidate_exclusion"] is False


def test_drawer_execution_terminal_failure_skips_post_action_verification() -> None:
    machine = BehaviorExecutionStateMachine()
    machine.start(drawer_pre_action_candidate(), now=0.0)
    machine.on_interaction_observation_result(
        {
            "attribute_status": "ready",
            "attribute_source": "mllm_attribute_inference",
            "is_currently_visible": True,
            "view_state": "front",
            "front_surface_visible": True,
            "approach_ready": True,
            "drawer_action_regions_ready": True,
            "attribute_capture_step": 1,
        },
        now=1.0,
    )
    terminal = machine.on_interaction_result(
        False,
        {
            "reason": "drawer_interaction_execution_unavailable",
            "failure_stage": "interaction_execution",
        },
        now=2.0,
    )
    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is False
    assert machine.state != STATE_VERIFYING


def test_real_interaction_backend_failure_still_requests_post_action_verification() -> None:
    machine = BehaviorExecutionStateMachine()
    machine.start(interaction_candidate(requires_approach=False), now=0.0)

    commands = machine.on_interaction_result(
        False, {"reason": "force_no_effect"}, now=1.0
    )
    assert machine.state == STATE_VERIFYING
    assert commands[0]["kind"] == "verify_interaction"
    assert commands[0]["backend_success"] is False

    terminal = machine.on_verification_result(
        True, {"m3_state": "open", "verified": True}, now=2.0
    )
    assert machine.state != STATE_SUCCEEDED
    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is False
    assert terminal[0]["detail"]["reason"] == "interaction_backend_failed"


def test_static_portal_feedback_finishes_directly_without_graph_or_timeout() -> None:
    machine = BehaviorExecutionStateMachine(
        ExecutionConfig(verification_timeout_s=0.01)
    )
    candidate = interaction_candidate(requires_approach=False)
    candidate["metadata"]["node_type"] = "portal"
    candidate["interaction_command"]["action"] = "open"

    commands = machine.start(candidate, now=0.0)
    assert commands[0]["kind"] == "interact"
    terminal = machine.on_interaction_result(
        True,
        {
            "status": "SUCCEEDED",
            "action": "open",
            "post_state": "static_open",
            # The direct executor route may omit interaction_capability; state
            # and source are independently sufficient public evidence.
            "source": "executor_static_portal",
        },
        now=1.0,
    )

    assert machine.state == STATE_SUCCEEDED
    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is True
    assert terminal[0]["detail"]["verification_mode"] == "direct_static_portal_feedback"
    assert machine.timeout_reason(now=100.0) == ""


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


def test_mandatory_scan_runs_in_executor_and_only_completion_succeeds() -> None:
    machine = BehaviorExecutionStateMachine(ExecutionConfig(scan_timeout_s=15.0))
    commands = machine.start(
        {
            "candidate_id": "startup_scan:episode_1",
            "behavior_type": "SCAN",
            "metadata": {"mandatory_startup_scan": True},
        },
        now=0.0,
    )

    assert machine.state == STATE_SCANNING
    assert commands == [
        {
            "kind": "scan",
            "candidate": {
                "candidate_id": "startup_scan:episode_1",
                "behavior_type": "SCAN",
                "metadata": {"mandatory_startup_scan": True},
            },
        }
    ]
    assert machine.timeout_reason(now=14.9) == ""
    assert machine.timeout_reason(now=15.1) == "scan_timeout"

    terminal = machine.on_scan_result(
        True,
        {"reason": "completed", "accumulated_yaw_rad": 6.29},
        now=15.2,
    )
    assert machine.state == STATE_SUCCEEDED
    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is True


def test_step_command_gate_pairs_callbacks_by_step_and_waits_for_ack() -> None:
    gate = StepCommandGate(max_pair_age_s=10.0)
    # ROS may deliver the gate before its RGB callback.
    gate.record_fresh_gate(11, now=1.0)
    assert gate.consume_step(now=1.0) is None
    gate.record_rgb(11, now=1.1)
    assert gate.consume_step(now=1.1) == 11
    # A later pair cannot be consumed until step 11 is acknowledged.
    gate.record_rgb(12, now=1.2)
    gate.record_fresh_gate(12, now=1.2)
    assert gate.consume_step(now=1.2) is None
    gate.record_step_sync(11, action_source="cmd_vel")
    ack = gate.take_acks()
    assert len(ack) == 1
    assert ack[0].command_applied
    assert gate.consume_step(now=1.3) == 12


def test_step_command_gate_retries_after_timeout_noop_without_reusing_stale_step() -> None:
    gate = StepCommandGate(max_pair_age_s=10.0)
    gate.record_rgb(21, now=2.0)
    gate.record_fresh_gate(21, now=2.0)
    assert gate.consume_step(now=2.0) == 21

    # A bridge action timeout acknowledges the evaluator step but must not
    # count as an applied turn. The next command can only use a new RGB/gate
    # pair, never the stale step 21 pair.
    gate.record_step_sync(21, action_source="timeout_noop")
    acknowledgement = gate.take_acks()
    assert len(acknowledgement) == 1
    assert not acknowledgement[0].command_applied
    assert gate.consume_step(now=2.1) is None

    gate.record_rgb(22, now=2.2)
    gate.record_fresh_gate(22, now=2.2)
    assert gate.consume_step(now=2.2) == 22


def test_step_command_gate_does_not_use_mismatched_latest_values() -> None:
    gate = StepCommandGate(max_pair_age_s=10.0)
    gate.record_rgb(20, now=1.0)
    gate.record_fresh_gate(19, now=1.0)
    assert gate.consume_step(now=1.0) is None
    gate.record_fresh_gate(20, now=1.1)
    assert gate.consume_step(now=1.1) == 20


def test_startup_scan_lifecycle_keeps_instance_stable_until_true_episode_change() -> None:
    lifecycle = StartupScanLifecycle(enabled=True)
    first_candidate_id = lifecycle.candidate_id

    # A map-ready stream with no graph id must still expose only SCAN.
    assert not lifecycle.should_publish_scan(False)
    assert lifecycle.should_publish_scan(True)
    assert not lifecycle.observe_episode("")
    assert lifecycle.candidate_id == first_candidate_id

    lifecycle.record_feedback(first_candidate_id, "STARTED")
    # The first non-empty graph id binds the existing scan; it is not a reset.
    assert not lifecycle.observe_episode("episode_000002")
    assert lifecycle.bound_episode_id == "episode_000002"
    assert lifecycle.candidate_id == first_candidate_id
    assert lifecycle.state == "ACTIVE"

    lifecycle.record_feedback(first_candidate_id, "SUCCEEDED")
    assert lifecycle.state == "COMPLETE"
    # Only a distinct later episode creates a second mandatory scan instance.
    assert lifecycle.observe_episode("episode_000003")
    assert lifecycle.candidate_id != first_candidate_id
    assert lifecycle.state == "PENDING"


def test_startup_scan_timeout_uses_control_steps_not_wall_clock() -> None:
    # 27 healthy acknowledgements may take >15 wall seconds on a loaded
    # simulator, but represent only 5.4 seconds of evaluator control time.
    assert startup_scan_elapsed_control_s(27, 0.2) == 5.4
    assert startup_scan_timeout_reason(
        acknowledged_control_steps=27,
        control_dt_s=0.2,
        timeout_s=15.0,
        awaiting_ack_step=None,
        last_command_sent_monotonic_s=None,
        now_monotonic_s=100.0,
        step_sync_stall_timeout_s=5.0,
    ) == ""
    assert startup_scan_timeout_reason(
        acknowledged_control_steps=75,
        control_dt_s=0.2,
        timeout_s=15.0,
        awaiting_ack_step=None,
        last_command_sent_monotonic_s=None,
        now_monotonic_s=100.0,
        step_sync_stall_timeout_s=5.0,
    ) == "scan_timeout"


def test_startup_scan_reports_step_sync_stall_separately() -> None:
    assert startup_scan_timeout_reason(
        acknowledged_control_steps=2,
        control_dt_s=0.2,
        timeout_s=15.0,
        awaiting_ack_step=9,
        last_command_sent_monotonic_s=10.0,
        now_monotonic_s=15.1,
        step_sync_stall_timeout_s=5.0,
    ) == "scan_step_sync_stall"
