from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SCRIPTS))

from semantic_decision_py_pkg.external_recovery import (  # noqa: E402
    RECOVERY_REQUEST_SOURCE,
    SAFE_REVERSE_REPLAN,
    recovery_feedback_allows_retry,
    normalize_recovery_request,
    recovery_request_matches_selection,
)


def _request(**overrides):
    payload = {
        "request_id": "decision-1:frontier-1:recovery:001",
        "source": RECOVERY_REQUEST_SOURCE,
        "recovery": SAFE_REVERSE_REPLAN,
        "decision_id": "decision-1",
        "candidate_id": "frontier-1",
        "cluster_id": "cluster-1",
        "goal_xyyaw": [1.0, -2.0, 0.5],
        "reason": "reserve_make_plan_unreachable",
        "timestamp": 12.0,
    }
    payload.update(overrides)
    return payload


def test_normalized_request_is_command_neutral_and_preserves_identity():
    request = normalize_recovery_request(
        _request(cmd_vel={"linear_x": -0.2}, controller_owner="explore_py")
    )

    assert request == {
        "request_id": "decision-1:frontier-1:recovery:001",
        "source": "explore_py",
        "recovery": "safe_reverse_replan",
        "decision_id": "decision-1",
        "candidate_id": "frontier-1",
        "cluster_id": "cluster-1",
        "goal_xyyaw": [1.0, -2.0, 0.5],
        "reason": "reserve_make_plan_unreachable",
        "timestamp": 12.0,
    }
    assert "cmd_vel" not in request
    assert "controller_owner" not in request


def test_request_requires_active_explore_identity_not_only_decision_id():
    request = normalize_recovery_request(_request())

    assert recovery_request_matches_selection(
        request,
        {
            "behavior_type": "EXPLORE",
            "decision_id": "decision-1",
            "candidate_id": "frontier-1",
        },
    )
    assert not recovery_request_matches_selection(
        request,
        {
            "behavior_type": "EXPLORE",
            "decision_id": "decision-1",
            "candidate_id": "frontier-2",
        },
    )
    assert not recovery_request_matches_selection(
        request,
        {
            "behavior_type": "INTERACT",
            "decision_id": "decision-1",
            "candidate_id": "frontier-1",
        },
    )


def test_rejects_invalid_source_kind_or_goal():
    assert normalize_recovery_request(_request(source="other")) is None
    assert normalize_recovery_request(_request(recovery="drive_reverse")) is None
    assert normalize_recovery_request(_request(goal_xyyaw=[float("nan"), 0.0])) is None
    assert normalize_recovery_request(_request(candidate_id="")) is None


def test_only_transient_rejection_allows_one_bounded_retry():
    transient = {
        "status": "REJECTED",
        "detail": {"reason": "recovery_request_wrong_executor_state"},
    }
    assert recovery_feedback_allows_retry(transient, retry_count=0, retry_limit=1)
    assert not recovery_feedback_allows_retry(transient, retry_count=1, retry_limit=1)
    assert recovery_feedback_allows_retry(
        {
            "status": "REJECTED",
            "detail": {"reason": "recovery_request_not_active_explore"},
        },
        retry_count=0,
        retry_limit=1,
    )
    assert not recovery_feedback_allows_retry(
        {"status": "REJECTED", "detail": {"reason": "external_recovery_disabled"}},
        retry_count=0,
        retry_limit=1,
    )
