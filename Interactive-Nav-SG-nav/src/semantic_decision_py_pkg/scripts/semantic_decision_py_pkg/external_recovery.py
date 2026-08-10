"""Small, command-neutral contract for explorer recovery requests.

The explorer can report that a reserved frontier repeatedly failed planning,
but it must not publish a competing velocity command.  This module keeps the
wire contract and matching rules independent of ROS so both sides can test the
same seam without importing the full executor.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


SAFE_REVERSE_REPLAN = "safe_reverse_replan"
RECOVERY_REQUEST_SOURCE = "explore_py"
TRANSIENT_RECOVERY_REJECTION_REASONS = frozenset(
    {
        "recovery_request_not_active_explore",
        "recovery_request_wrong_executor_state",
    }
)


def _finite_goal(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        values = [float(value[0]), float(value[1]), float(value[2]) if len(value) > 2 else 0.0]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in values):
        return None
    return values


def normalize_recovery_request(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate and normalize one explorer recovery request.

    Unknown fields are intentionally discarded.  In particular, a request is
    never allowed to carry a velocity or controller-owner field across this
    boundary; the executor chooses its existing safe-reverse implementation.
    """

    if not isinstance(payload, Mapping):
        return None
    if str(payload.get("source") or "").strip().casefold() != RECOVERY_REQUEST_SOURCE:
        return None
    if str(payload.get("recovery") or "").strip().casefold() != SAFE_REVERSE_REPLAN:
        return None
    request_id = str(payload.get("request_id") or "").strip()
    decision_id = str(payload.get("decision_id") or "").strip()
    candidate_id = str(payload.get("candidate_id") or "").strip()
    if not request_id or not decision_id or not candidate_id:
        return None
    goal = _finite_goal(payload.get("goal_xyyaw"))
    if goal is None:
        return None
    try:
        timestamp = float(payload.get("timestamp", 0.0) or 0.0)
    except (TypeError, ValueError):
        timestamp = 0.0
    if not math.isfinite(timestamp):
        timestamp = 0.0
    return {
        "request_id": request_id[:128],
        "source": RECOVERY_REQUEST_SOURCE,
        "recovery": SAFE_REVERSE_REPLAN,
        "decision_id": decision_id[:128],
        "candidate_id": candidate_id[:128],
        "cluster_id": str(payload.get("cluster_id") or "")[:128],
        "goal_xyyaw": goal,
        "reason": str(payload.get("reason") or "make_plan_unreachable")[:240],
        "timestamp": timestamp,
    }


def recovery_request_matches_selection(
    request: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None,
) -> bool:
    """Return whether a normalized request belongs to the active EXPLORE goal."""

    if not isinstance(request, Mapping) or not isinstance(selection, Mapping):
        return False
    if str(selection.get("behavior_type") or "").strip().upper() != "EXPLORE":
        return False
    return bool(
        str(request.get("decision_id") or "") == str(selection.get("decision_id") or "")
        and str(request.get("candidate_id") or "") == str(selection.get("candidate_id") or "")
    )


def recovery_feedback_allows_retry(
    payload: Mapping[str, Any] | None,
    *,
    retry_count: int,
    retry_limit: int,
) -> bool:
    """Allow one bounded resend when ROS delivered before executor readiness.

    A rejected recovery request normally means that its selected EXPLORE command
    is no longer valid.  The two explicit reasons below are different: they can
    occur while the executor and explorer are receiving the same reservation
    transition on separate ROS callbacks.  Retrying is deliberately bounded and
    does not grant ExplorePy any motion authority.
    """

    if int(retry_count) >= max(0, int(retry_limit)) or not isinstance(payload, Mapping):
        return False
    if str(payload.get("status") or "").strip().upper() != "REJECTED":
        return False
    detail = payload.get("detail")
    reason = str(detail.get("reason") or "") if isinstance(detail, Mapping) else ""
    return reason in TRANSIENT_RECOVERY_REJECTION_REASONS
