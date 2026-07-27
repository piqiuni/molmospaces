from semantic_decision_py_pkg.mission_completion import (
    MissionCompletionConfig,
    MissionCompletionTracker,
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
