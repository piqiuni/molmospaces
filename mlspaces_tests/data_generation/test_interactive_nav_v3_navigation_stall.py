from __future__ import annotations

from scripts.InteractiveNav.evaluation.ros_navigation_stall import (
    CROSS_SUBGOAL_STALL_REASON,
    CrossSubgoalStallConfig,
    CrossSubgoalStallTracker,
)


def _failed_feedback(index: int, *, behavior_type: str = "EXPLORE") -> dict[str, object]:
    return {
        "decision_id": f"decision-{index}",
        "candidate_id": f"candidate-{index}",
        "behavior_type": behavior_type,
        "status": "FAILED",
        "success": False,
        "detail": {"reason": "move_base_failed"},
    }


def test_cross_subgoal_failures_without_motion_trigger_early_stop() -> None:
    tracker = CrossSubgoalStallTracker(
        CrossSubgoalStallConfig(
            min_failed_subgoals=3,
            max_displacement_m=0.15,
            min_no_progress_steps=5,
        )
    )
    tracker.observe_pose((0.0, 0.0), 0)

    assert not tracker.note_feedback(_failed_feedback(1), (0.04, 0.02), 5)
    assert not tracker.note_feedback(_failed_feedback(2), (0.04, 0.02), 6)
    assert tracker.note_feedback(_failed_feedback(3), (0.04, 0.02), 7)

    snapshot = tracker.snapshot()
    assert snapshot["triggered"] is True
    assert snapshot["reason"] == CROSS_SUBGOAL_STALL_REASON
    assert snapshot["failed_subgoal_count"] == 3
    assert snapshot["observed_navigation_failure_count"] == 3
    assert snapshot["displacement_m"] is not None
    assert float(snapshot["displacement_m"]) < 0.15


def test_duplicate_feedback_and_real_motion_do_not_fabricate_cross_subgoal_stall() -> None:
    tracker = CrossSubgoalStallTracker(
        CrossSubgoalStallConfig(
            min_failed_subgoals=2,
            max_displacement_m=0.15,
            min_no_progress_steps=0,
        )
    )
    tracker.observe_pose((0.0, 0.0), 0)
    first = _failed_feedback(1, behavior_type="NAVIGATE")

    assert not tracker.note_feedback(first, (0.0, 0.0), 1)
    assert not tracker.note_feedback(first, (0.0, 0.0), 2)
    assert tracker.snapshot()["failed_subgoal_count"] == 1

    # Crossing the displacement threshold resets the run before the second
    # failure: this is recovery progress, not a persistent lockup.
    assert not tracker.note_feedback(_failed_feedback(2), (0.20, 0.0), 3)
    snapshot = tracker.snapshot()
    assert snapshot["triggered"] is False
    assert snapshot["failed_subgoal_count"] == 1
    assert snapshot["observed_navigation_failure_count"] == 2


def test_successful_navigation_resets_prior_failure_sequence() -> None:
    tracker = CrossSubgoalStallTracker(
        CrossSubgoalStallConfig(
            min_failed_subgoals=2,
            max_displacement_m=0.15,
            min_no_progress_steps=0,
        )
    )
    tracker.observe_pose((0.0, 0.0), 0)
    assert not tracker.note_feedback(_failed_feedback(1), (0.0, 0.0), 1)
    assert not tracker.note_feedback(
        {
            "decision_id": "decision-success",
            "behavior_type": "NAVIGATE",
            "status": "SUCCEEDED",
            "success": True,
        },
        (0.0, 0.0),
        2,
    )
    assert not tracker.note_feedback(_failed_feedback(2), (0.0, 0.0), 3)
    assert tracker.snapshot()["failed_subgoal_count"] == 1
