from semantic_decision_py_pkg.mission_completion import (
    InteractionApproachFailureLimitConfig,
    InteractionApproachFailureLimitTracker,
    MissionCompletionConfig,
    MissionCompletionTracker,
    TerminalInteractionNoPlanExitConfig,
    TerminalInteractionNoPlanExitTracker,
    TargetMissionTracker,
)
from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate


def payload(sequence: int, *, exhausted: bool, candidate_count: int) -> dict:
    return {
        "sequence": sequence,
        "candidate_count": candidate_count,
        "exploration_context": {"frontier_exhausted": exhausted},
    }


def test_completion_requires_distinct_stable_empty_sequences() -> None:
    tracker = MissionCompletionTracker(
        MissionCompletionConfig(empty_candidate_confirmations=2)
    )

    assert tracker.update(
        payload(1, exhausted=True, candidate_count=0),
        has_active_behavior=False,
        target_enabled=False,
    ) is False
    assert tracker.update(
        payload(1, exhausted=True, candidate_count=0),
        has_active_behavior=False,
        target_enabled=False,
    ) is False
    assert tracker.update(
        payload(2, exhausted=True, candidate_count=0),
        has_active_behavior=False,
        target_enabled=False,
    ) is True


def test_completion_uses_navigation_and_interaction_frontiers_with_target_enabled() -> None:
    tracker = MissionCompletionTracker(
        MissionCompletionConfig(empty_candidate_confirmations=1)
    )

    assert tracker.update(
        payload(1, exhausted=True, candidate_count=0),
        has_active_behavior=False,
        target_enabled=True,
    ) is True


def test_completion_is_blocked_by_actions_or_interaction_frontiers() -> None:
    tracker = MissionCompletionTracker(
        MissionCompletionConfig(empty_candidate_confirmations=1)
    )

    assert tracker.update(
        payload(1, exhausted=True, candidate_count=0),
        has_active_behavior=True,
        target_enabled=False,
    ) is False
    assert tracker.update(
        {
            "sequence": 2,
            "candidate_count": 1,
            "candidates": [
                {
                    "behavior_type": "INTERACT",
                    "metadata": {"interaction_group_already_explored": False},
                }
            ],
            "exploration_context": {
                "frontier_exhausted": False,
                "navigation_frontier_exhausted": True,
                "navigation_frontier_count": 0,
                "interaction_frontier_exhausted": False,
                "interaction_frontier_count": 1,
                "combined_frontier_count": 1,
            },
        },
        has_active_behavior=False,
        target_enabled=False,
    ) is False


def test_stagnation_requests_recovery_instead_of_completing_mission() -> None:
    tracker = MissionCompletionTracker(
        MissionCompletionConfig(
            empty_candidate_confirmations=3,
            stagnation_failure_limit=2,
        )
    )
    tracker.note_feedback(
        {"behavior_type": "EXPLORE", "status": "FAILED", "detail": {}}
    )
    tracker.note_feedback(
        {"behavior_type": "EXPLORE", "status": "FAILED", "detail": {}}
    )
    assert tracker.update(
        {
            "sequence": 1,
            "candidate_count": 1,
            "exploration_context": {"frontier_exhausted": False},
        },
        has_active_behavior=False,
        target_enabled=False,
    ) is False
    assert tracker.complete is False
    assert tracker.stalled is True
    assert tracker.reason == "exploration_stalled_recovery"
    assert tracker.failure_streak == 0


def test_completion_waits_for_initial_scan() -> None:
    tracker = MissionCompletionTracker(
        MissionCompletionConfig(empty_candidate_confirmations=1)
    )
    pending_scan = payload(1, exhausted=True, candidate_count=0)
    pending_scan["exploration_context"]["initial_scan_complete"] = False

    assert tracker.update(
        pending_scan,
        has_active_behavior=False,
        target_enabled=False,
    ) is False


def test_completion_requires_fifty_observation_steps_after_frontiers_empty() -> None:
    tracker = MissionCompletionTracker(
        MissionCompletionConfig(
            empty_candidate_confirmations=1,
            empty_candidate_min_steps=50,
        )
    )

    first = payload(1, exhausted=True, candidate_count=0)
    first["exploration_context"]["observation_step"] = 100
    assert tracker.update(
        first, has_active_behavior=False, target_enabled=False
    ) is False

    before_limit = payload(2, exhausted=True, candidate_count=0)
    before_limit["exploration_context"]["observation_step"] = 149
    assert tracker.update(
        before_limit, has_active_behavior=False, target_enabled=False
    ) is False

    at_limit = payload(3, exhausted=True, candidate_count=0)
    at_limit["exploration_context"]["observation_step"] = 150
    assert tracker.update(
        at_limit, has_active_behavior=False, target_enabled=False
    ) is True


def _no_executable_snapshot(sequence: int, observation_step: int) -> dict:
    return {
        "sequence": sequence,
        # The raw producer may continue to advertise a terminal interaction,
        # while episode-local reachability memory makes the eligible set empty.
        "candidate_count": 1,
        "candidates": [{"candidate_id": "interaction:drawer:open"}],
        "exploration_context": {
            "initial_scan_complete": True,
            "observation_step": observation_step,
        },
    }


def _terminal_interaction_no_plan_feedback() -> dict:
    return {
        "status": "FAILED",
        "behavior_type": "INTERACT",
        "candidate_id": "interaction:drawer:open",
        "decision_id": "decision_000007",
        "detail": {
            "reason": InteractionApproachFailureLimitTracker.REASON,
            "interaction_failure_count": 3,
            "interaction_failure_limit": 3,
        },
    }


def test_interaction_approach_failures_become_terminal_only_at_limit() -> None:
    tracker = InteractionApproachFailureLimitTracker(
        InteractionApproachFailureLimitConfig(failure_limit=3)
    )
    candidate_id = "interaction:container_drawer:open"

    # Action/verification failures are not navigation approach failures.
    assert tracker.note_feedback(
        candidate_id=candidate_id,
        behavior_type="INTERACT",
        status="FAILED",
        failure_stage="interaction_verification",
    ) is None
    assert tracker.failure_counts == {}

    assert tracker.note_feedback(
        candidate_id=candidate_id,
        behavior_type="INTERACT",
        status="FAILED",
        failure_stage="interaction_approach_navigation",
    ) is None
    assert tracker.note_feedback(
        candidate_id="interaction:other:open",
        behavior_type="INTERACT",
        status="REJECTED",
        failure_stage="interaction_approach_navigation",
    ) is None
    assert tracker.note_feedback(
        candidate_id=candidate_id,
        behavior_type="INTERACT",
        status="REJECTED",
        failure_stage="interaction_approach_navigation",
    ) is None

    terminal = tracker.note_feedback(
        candidate_id=candidate_id,
        behavior_type="INTERACT",
        status="FAILED",
        failure_stage="interaction_approach_navigation",
    )
    assert terminal == {
        "reason": InteractionApproachFailureLimitTracker.REASON,
        "interaction_failure_count": 3,
        "interaction_failure_limit": 3,
        "terminal_candidate_exclusion": True,
    }
    assert tracker.terminal_candidate_ids == {candidate_id}
    assert tracker.failure_counts["interaction:other:open"] == 1

    # Terminal means terminal for this episode: duplicate late success feedback
    # cannot make the same candidate selectable again.
    assert tracker.note_feedback(
        candidate_id=candidate_id,
        behavior_type="INTERACT",
        status="SUCCEEDED",
    ) is None
    assert tracker.terminal_candidate_ids == {candidate_id}
    tracker.reset()
    assert tracker.terminal_candidate_ids == set()


def test_visual_and_execution_terminal_feedback_exclude_candidate_immediately() -> None:
    tracker = InteractionApproachFailureLimitTracker(
        InteractionApproachFailureLimitConfig(failure_limit=3)
    )
    visual = tracker.note_feedback(
        candidate_id="interaction:drawer:open",
        behavior_type="INTERACT",
        status="FAILED",
        failure_stage="interaction_visual_precondition",
    )
    assert visual["reason"] == (
        InteractionApproachFailureLimitTracker.VISUAL_PRECONDITION_REASON
    )
    assert tracker.terminal_candidate_ids == {"interaction:drawer:open"}

    tracker.reset()
    execution = tracker.note_feedback(
        candidate_id="interaction:drawer:open",
        behavior_type="INTERACT",
        status="REJECTED",
        failure_stage="interaction_execution",
    )
    assert execution["reason"] == (
        InteractionApproachFailureLimitTracker.EXECUTION_REASON
    )
    assert execution["interaction_failure_limit"] == 1


def test_all_interaction_approach_poses_exhausted_is_terminal_immediately() -> None:
    tracker = InteractionApproachFailureLimitTracker(
        InteractionApproachFailureLimitConfig(failure_limit=3)
    )

    terminal = tracker.note_feedback(
        candidate_id="interaction:fridge:open",
        behavior_type="INTERACT",
        status="FAILED",
        failure_stage=InteractionApproachFailureLimitTracker.APPROACH_EXHAUSTED_STAGE,
    )

    assert terminal is not None
    assert terminal["reason"] == InteractionApproachFailureLimitTracker.REASON
    assert terminal["interaction_approach_options_exhausted"] is True
    assert terminal["terminal_candidate_exclusion"] is True
    assert tracker.terminal_candidate_ids == {"interaction:fridge:open"}


def test_raw_frontier_candidate_prevents_false_exhaustion_from_stale_counts() -> None:
    tracker = MissionCompletionTracker(
        MissionCompletionConfig(empty_candidate_confirmations=1, empty_candidate_min_steps=0)
    )
    snapshot = {
        "sequence": 1,
        "candidate_count": 1,
        "candidates": [{"candidate_id": "frontier:1:2", "behavior_type": "EXPLORE"}],
        "exploration_context": {
            "initial_scan_complete": True,
            "observation_step": 10,
            # Simulate a producer counter reset racing one still-live raw
            # frontier candidate.
            "frontier_exhausted": True,
            "navigation_frontier_exhausted": True,
            "navigation_frontier_count": 0,
            "interaction_frontier_exhausted": True,
            "interaction_frontier_count": 0,
            "combined_frontier_count": 0,
        },
    }

    assert not tracker.update(
        snapshot, has_active_behavior=False, target_enabled=False
    )
    assert tracker.complete is False


def test_material_filtered_frontier_blocks_exhaustion_then_reports_bounded_stall() -> None:
    tracker = MissionCompletionTracker(
        MissionCompletionConfig(
            empty_candidate_confirmations=1,
            empty_candidate_min_steps=0,
            retryable_frontier_stall_min_steps=20,
            retryable_frontier_stall_confirmations=3,
        )
    )

    def snapshot(sequence: int, observation_step: int) -> dict:
        return {
            "sequence": sequence,
            "candidate_count": 0,
            "candidates": [],
            "exploration_context": {
                "initial_scan_complete": True,
                "observation_step": observation_step,
                # The candidate generator has no safe viewpoint yet, but the
                # explorer still has a material frontier region to re-evaluate.
                "frontier_exhausted": False,
                "navigation_frontier_exhausted": False,
                "interaction_frontier_exhausted": True,
                "filtered_frontier_retryable": True,
                "filtered_frontier_reason": "material_frontier_without_safe_viewpoint",
                "raw_frontier_cluster_count": 2,
                "raw_frontier_material_cluster_count": 1,
                "filtered_no_viewpoint_frontier_cluster_count": 1,
            },
        }

    assert not tracker.update(
        snapshot(1, 100), has_active_behavior=False, target_enabled=False
    )
    assert not tracker.update(
        snapshot(2, 110), has_active_behavior=False, target_enabled=False
    )
    assert not tracker.update(
        snapshot(3, 120), has_active_behavior=False, target_enabled=False
    )
    assert tracker.complete is False
    assert tracker.terminal_stalled is True
    assert tracker.reason == "material_frontier_without_safe_viewpoint"
    assert tracker.last_retryable_frontier_detail[
        "retryable_frontier_elapsed_steps"
    ] == 20


def test_tiny_only_frontier_does_not_trigger_retryable_terminal_guard() -> None:
    tracker = MissionCompletionTracker(
        MissionCompletionConfig(empty_candidate_confirmations=1, empty_candidate_min_steps=0)
    )
    snapshot = {
        "sequence": 1,
        "candidate_count": 0,
        "candidates": [],
        "exploration_context": {
            "initial_scan_complete": True,
            "observation_step": 10,
            "frontier_exhausted": True,
            "navigation_frontier_exhausted": True,
            "interaction_frontier_exhausted": True,
            # A tiny-only filter result must remain a normal terminal state.
            "filtered_frontier_retryable": False,
            "raw_frontier_cluster_count": 1,
            "raw_frontier_material_cluster_count": 0,
            "filtered_tiny_frontier_cluster_count": 1,
        },
    }

    assert tracker.update(snapshot, has_active_behavior=False, target_enabled=False)
    assert tracker.complete is True
    assert tracker.terminal_stalled is False


def test_completion_is_blocked_by_connected_unknown_area_without_candidate() -> None:
    tracker = MissionCompletionTracker(
        MissionCompletionConfig(empty_candidate_confirmations=1, empty_candidate_min_steps=0)
    )
    snapshot = payload(1, exhausted=True, candidate_count=0)
    snapshot["exploration_context"].update(
        {
            "initial_scan_complete": True,
            "observation_step": 10,
            "connected_unknown_area_present": True,
            "navigation_frontier_exhausted": True,
            "interaction_frontier_exhausted": True,
            "combined_frontier_count": 0,
        }
    )
    assert not tracker.update(snapshot, has_active_behavior=False, target_enabled=False)
    assert tracker.complete is False


def test_completion_is_blocked_by_remembered_unresolved_interaction() -> None:
    tracker = MissionCompletionTracker(
        MissionCompletionConfig(empty_candidate_confirmations=1, empty_candidate_min_steps=0)
    )
    snapshot = payload(1, exhausted=True, candidate_count=0)
    snapshot["exploration_context"].update(
        {
            "initial_scan_complete": True,
            "observation_step": 10,
            "navigation_frontier_exhausted": True,
            "interaction_frontier_exhausted": True,
            "combined_frontier_count": 0,
        }
    )
    snapshot["graph_context"] = {
        "nodes": [
            {
                "id": "door_0003",
                "type": "portal",
                "requires_interaction": True,
                "interaction_state": "closed",
                "interaction_capability": "unknown",
                "is_currently_visible": False,
            }
        ]
    }
    assert not tracker.update(snapshot, has_active_behavior=False, target_enabled=False)
    assert tracker.complete is False


def test_completion_is_blocked_while_interaction_target_is_on_cooldown() -> None:
    tracker = MissionCompletionTracker(
        MissionCompletionConfig(empty_candidate_confirmations=1, empty_candidate_min_steps=0)
    )
    snapshot = payload(1, exhausted=True, candidate_count=0)
    snapshot["exploration_context"].update(
        {
            "initial_scan_complete": True,
            "observation_step": 10,
            "navigation_frontier_exhausted": True,
            "interaction_frontier_exhausted": True,
            "combined_frontier_count": 0,
            "interaction_cooldown_target_count": 1,
        }
    )
    assert not tracker.update(snapshot, has_active_behavior=False, target_enabled=False)
    assert tracker.complete is False


def test_single_make_plan_failure_does_not_bypass_approach_failure_limit() -> None:
    tracker = TerminalInteractionNoPlanExitTracker(
        TerminalInteractionNoPlanExitConfig(enabled=True)
    )
    assert not tracker.note_feedback(
        {
            "status": "FAILED",
            "behavior_type": "INTERACT",
            "candidate_id": "interaction:drawer:open",
            "detail": {"reason": "make_plan_unreachable"},
        },
        observation_step=10,
    )
    assert tracker.terminal_failure == {}


def test_terminal_no_plan_tracker_accepts_visual_and_execution_reasons() -> None:
    tracker = TerminalInteractionNoPlanExitTracker(
        TerminalInteractionNoPlanExitConfig(enabled=True)
    )
    for reason in (
        InteractionApproachFailureLimitTracker.VISUAL_PRECONDITION_REASON,
        InteractionApproachFailureLimitTracker.EXECUTION_REASON,
    ):
        assert tracker.note_feedback(
            {
                "status": "FAILED",
                "behavior_type": "INTERACT",
                "candidate_id": "interaction:drawer:open",
                "detail": {"reason": reason},
            },
            observation_step=10,
        )
        tracker.reset()


def test_terminal_interaction_no_plan_exits_only_after_distinct_observation_steps() -> None:
    tracker = TerminalInteractionNoPlanExitTracker(
        TerminalInteractionNoPlanExitConfig(
            enabled=True,
            no_executable_candidate_min_steps=20,
            no_executable_candidate_confirmations=3,
        )
    )
    assert tracker.note_feedback(
        _terminal_interaction_no_plan_feedback(), observation_step=100
    )

    assert not tracker.update(
        _no_executable_snapshot(1, 101),
        has_active_behavior=False,
        has_executable_candidate=False,
        startup_scan_pending=False,
        eligible_candidate_count=0,
    )
    # Multiple ROS messages from the same simulation observation must not
    # synthesize a no-candidate streak.
    assert not tracker.update(
        _no_executable_snapshot(2, 101),
        has_active_behavior=False,
        has_executable_candidate=False,
        startup_scan_pending=False,
        eligible_candidate_count=0,
    )
    assert not tracker.update(
        _no_executable_snapshot(3, 119),
        has_active_behavior=False,
        has_executable_candidate=False,
        startup_scan_pending=False,
        eligible_candidate_count=0,
    )
    assert tracker.update(
        _no_executable_snapshot(4, 120),
        has_active_behavior=False,
        has_executable_candidate=False,
        startup_scan_pending=False,
        eligible_candidate_count=0,
    )
    assert tracker.reason == (
        "no_executable_candidates_after_terminal_interaction_no_plan"
    )
    assert tracker.last_detail["raw_candidate_count"] == 1
    assert tracker.last_detail["eligible_candidate_count"] == 0
    assert tracker.last_detail["terminal_interaction_failure"]["detail"]["reason"] == (
        InteractionApproachFailureLimitTracker.REASON
    )


def test_terminal_interaction_no_plan_preserves_scan_and_recovery_paths() -> None:
    tracker = TerminalInteractionNoPlanExitTracker(
        TerminalInteractionNoPlanExitConfig(
            enabled=True,
            no_executable_candidate_min_steps=1,
            no_executable_candidate_confirmations=1,
        )
    )
    assert tracker.note_feedback(
        _terminal_interaction_no_plan_feedback(), observation_step=10
    )
    assert not tracker.update(
        _no_executable_snapshot(1, 11),
        has_active_behavior=False,
        has_executable_candidate=False,
        startup_scan_pending=True,
        eligible_candidate_count=0,
    )
    assert tracker.terminal_failure == {}

    assert tracker.note_feedback(
        _terminal_interaction_no_plan_feedback(), observation_step=20
    )
    assert not tracker.update(
        _no_executable_snapshot(2, 21),
        has_active_behavior=False,
        has_executable_candidate=True,
        startup_scan_pending=False,
        eligible_candidate_count=1,
    )
    assert tracker.terminal_failure == {}
    # A past failed interaction cannot terminate a later recovery cycle.
    assert not tracker.update(
        _no_executable_snapshot(3, 40),
        has_active_behavior=False,
        has_executable_candidate=False,
        startup_scan_pending=False,
        eligible_candidate_count=0,
    )


def test_target_mission_requires_matching_interaction_after_navigation() -> None:
    tracker = TargetMissionTracker()
    transition = tracker.on_behavior_succeeded(
        behavior_type="NAVIGATE",
        active_target_goal=True,
        target_context={"require_interaction": True},
        feedback={"target_id": "container_fridge", "target_name": "fridge"},
        next_candidate_sequence=8,
    )
    assert transition["phase"] == "target_reached"
    assert transition["minimum_candidate_sequence"] == 8

    candidates = [
        BehaviorCandidate(
            candidate_id="interaction:other:open",
            behavior_type="INTERACT",
            source="test",
            target_id="other",
            target_name="cabinet",
        ),
        BehaviorCandidate(
            candidate_id="interaction:fridge:open",
            behavior_type="INTERACT",
            source="test",
            target_id="container_fridge",
            target_name="fridge",
        ),
    ]
    filtered = tracker.filter_candidates(candidates)
    assert [candidate.candidate_id for candidate in filtered] == [
        "interaction:fridge:open"
    ]

    transition = tracker.on_behavior_succeeded(
        behavior_type="INTERACT",
        active_target_goal=True,
        target_context={"require_interaction": True},
        feedback={"detail": {"state": "open"}},
        next_candidate_sequence=9,
    )
    assert transition["phase"] == "complete"
    assert transition["detail"]["target_interaction_complete"] is True
    assert tracker.pending_interaction is None


def test_target_mission_accepts_matching_autonomous_interaction() -> None:
    tracker = TargetMissionTracker()
    target_context = {
        "enabled": True,
        "target_name": "fridge",
        "object_labels": ["fridge", "refrigerator"],
        "require_interaction": True,
    }
    feedback = {
        "target_id": "container_gt_000019",
        "target_name": "refrigerator_4d8cd69ca487b76cae801cfb0248a055_1_0_6",
    }
    candidates = [
        {
            "target_id": "container_gt_000019",
            "target_name": feedback["target_name"],
            "metadata": {"target_goal": True},
        }
    ]

    assert tracker.matches_target_interaction(
        target_context=target_context,
        feedback=feedback,
        candidates=candidates,
    ) is True
    transition = tracker.on_behavior_succeeded(
        behavior_type="INTERACT",
        active_target_goal=True,
        target_context=target_context,
        feedback={"detail": {"state": "open"}, **feedback},
        next_candidate_sequence=4,
    )
    assert transition["phase"] == "complete"
    assert transition["detail"]["target_interaction_complete"] is True


def test_target_mission_rejects_unrelated_autonomous_interaction() -> None:
    tracker = TargetMissionTracker()
    assert tracker.matches_target_interaction(
        target_context={
            "enabled": True,
            "target_name": "fridge",
            "object_labels": ["refrigerator"],
        },
        feedback={"target_id": "container_cabinet", "target_name": "cabinet_1"},
        candidates=[],
    ) is False


def test_visible_object_goal_becomes_priority_navigation_candidate() -> None:
    candidate = TargetMissionTracker.priority_target_candidate(
        [
            {
                "candidate_id": "target:object_apple",
                "behavior_type": "NAVIGATE",
                "target_id": "object_apple",
                "target_name": "apple",
                "metadata": {
                    "target_goal": True,
                    "target_visible_now": True,
                    "target_reliably_observed": True,
                },
            }
        ]
    )

    assert candidate["candidate_id"] == "target:object_apple"
    assert candidate["behavior_type"] == "NAVIGATE"


def test_visible_only_priority_rejects_historical_target_observation() -> None:
    candidate = TargetMissionTracker.priority_target_candidate(
        [
            {
                "candidate_id": "target:object_apple",
                "behavior_type": "NAVIGATE",
                "metadata": {
                    "target_goal": True,
                    "target_visible_now": False,
                    "target_reliably_observed": True,
                },
            }
        ],
        currently_visible_only=True,
    )

    assert candidate is None


def test_reliable_historical_target_remains_priority_navigation_candidate() -> None:
    candidate = TargetMissionTracker.priority_target_candidate(
        [
            {
                "candidate_id": "target:object_lettuce",
                "behavior_type": "NAVIGATE",
                "metadata": {
                    "target_goal": True,
                    "target_visible_now": False,
                    "target_reliably_observed": True,
                },
            }
        ]
    )

    assert candidate is not None
    assert candidate["candidate_id"] == "target:object_lettuce"
