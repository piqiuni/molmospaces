from __future__ import annotations

from scripts.InteractiveNav.evaluation.goal_status import (
    GoalStatusObserver,
    PublicGoalEvidenceLedger,
    is_exploration_terminal,
    is_target_goal_success_claim,
    verify_target_goal_claim,
)


EPISODE_ID = "eval_000007"


def _status(
    *,
    episode_id: str = EPISODE_ID,
    status: str = "SUCCEEDED",
    reason: str = "target_goal_succeeded",
    mission_mode: str = "semantic_interaction_object_goal",
    timestamp: float = 100.0,
) -> dict:
    return {
        "status": status,
        "mission_mode": mission_mode,
        "target_context": {"episode_id": episode_id},
        "timestamp": timestamp,
        "detail": {"reason": reason},
    }


def _frame(*instance_ids: str, episode_id: str = EPISODE_ID) -> dict:
    return {
        "episode_id": episode_id,
        "capture_step": 12,
        "observations": [{"id": instance_id, "name": "object"} for instance_id in instance_ids],
    }


def test_goal_status_observer_rejects_latched_and_cross_episode_messages() -> None:
    now = [100.0]
    observer = GoalStatusObserver(clock=lambda: now[0], stale_tolerance_s=0.0)
    observer.begin_episode(EPISODE_ID)

    assert observer.ingest(_status(episode_id="eval_000006")) is False
    assert observer.ingest(_status(timestamp=99.9)) is False
    assert observer.ingest(_status(timestamp=100.0)) is True
    assert observer.drain() == [_status(timestamp=100.0)]
    assert observer.drain() == []


def test_goal_success_requires_specific_reason_and_object_goal_mode() -> None:
    assert is_target_goal_success_claim(_status()) is True
    assert is_target_goal_success_claim(_status(reason="exploration_exhausted")) is False
    assert is_target_goal_success_claim(_status(mission_mode="semantic_interaction_exploration")) is False
    assert is_exploration_terminal(
        _status(status="EXPLORATION_EXHAUSTED", reason="navigation_and_interaction_frontiers_exhausted")
    ) is True
    assert is_exploration_terminal(_status(reason="exploration_exhausted")) is False


def test_raw_private_visibility_without_published_target_does_not_verify() -> None:
    ledger = PublicGoalEvidenceLedger(
        episode_id=EPISODE_ID,
        target_instance_ids={"obj_target"},
    )
    # A different object was published; the simulator may still see one or two
    # raw target pixels, but those are below the restricted-frame reliability
    # gate and therefore absent here.
    ledger.record_frame(_frame("obj_container"), capture_step=12)

    result = verify_target_goal_claim(
        _status(),
        episode_id=EPISODE_ID,
        evidence=ledger,
        private_distances_m={"obj_target": 1.0},
        distance_threshold_m=1.5,
    )

    assert result.accepted is False
    assert result.reason == "no_published_target_evidence"


def test_claim_requires_matching_published_id_and_private_distance() -> None:
    ledger = PublicGoalEvidenceLedger(
        episode_id=EPISODE_ID,
        target_instance_ids={"obj_target"},
    )
    assert ledger.record_frame(
        _frame("obj_target", "obj_container"), capture_step=21
    ) == ("obj_target",)

    too_far = verify_target_goal_claim(
        _status(),
        episode_id=EPISODE_ID,
        evidence=ledger,
        private_distances_m={"obj_target": 1.5},
        distance_threshold_m=1.5,
    )
    accepted = verify_target_goal_claim(
        _status(),
        episode_id=EPISODE_ID,
        evidence=ledger,
        private_distances_m={"obj_target": 1.49},
        distance_threshold_m=1.5,
    )

    assert too_far.accepted is False
    assert too_far.reason == "private_distance_failed"
    assert accepted.accepted is True
    assert accepted.target_instance_id == "obj_target"
    assert accepted.evidence_capture_step == 21


def test_transient_drawer_frame_is_evidence_but_never_terminates_without_claim() -> None:
    ledger = PublicGoalEvidenceLedger(
        episode_id=EPISODE_ID,
        target_instance_ids={"obj_target"},
    )
    ledger.record_frame(_frame("obj_target"), capture_step=30)
    ledger.record_frame(_frame(), capture_step=31)

    assert ledger.has_reliable_target_evidence() is True
    # Recording evidence alone has no terminal API and cannot manufacture a
    # claim.  Once the policy explicitly declares completion, the same frame is
    # still valid even if the evaluator-owned drawer macro closed afterward.
    accepted = verify_target_goal_claim(
        _status(),
        episode_id=EPISODE_ID,
        evidence=ledger,
        private_distances_m={"obj_target": 1.0},
        distance_threshold_m=1.5,
    )
    assert accepted.accepted is True


def test_claim_requires_evidence_for_current_nearest_target_candidate() -> None:
    ledger = PublicGoalEvidenceLedger(
        episode_id=EPISODE_ID,
        target_instance_ids={"obj_old", "obj_nearest"},
    )
    ledger.record_frame(_frame("obj_old"), capture_step=18)

    rejected = verify_target_goal_claim(
        _status(),
        episode_id=EPISODE_ID,
        evidence=ledger,
        private_distances_m={"obj_old": 1.2, "obj_nearest": 0.8},
        distance_threshold_m=1.5,
    )
    assert rejected.accepted is False
    assert rejected.reason == "nearest_target_not_published"

    ledger.record_frame(_frame("obj_nearest"), capture_step=22)
    accepted = verify_target_goal_claim(
        _status(),
        episode_id=EPISODE_ID,
        evidence=ledger,
        private_distances_m={"obj_old": 1.2, "obj_nearest": 0.8},
        distance_threshold_m=1.5,
    )
    assert accepted.accepted is True
    assert accepted.target_instance_id == "obj_nearest"
    assert accepted.evidence_capture_step == 22


def test_equal_distance_target_candidates_preserve_native_candidate_order() -> None:
    ledger = PublicGoalEvidenceLedger(
        episode_id=EPISODE_ID,
        target_instance_ids={"obj_first", "obj_second"},
    )
    ledger.record_frame(_frame("obj_second"), capture_step=9)

    result = verify_target_goal_claim(
        _status(),
        episode_id=EPISODE_ID,
        evidence=ledger,
        private_distances_m={"obj_first": 1.0, "obj_second": 1.0},
        distance_threshold_m=1.5,
    )
    assert result.accepted is False
    assert result.reason == "nearest_target_not_published"
