from __future__ import annotations

import pytest

from scripts.InteractiveNav.evaluation import benchmark_metrics


def _episode() -> dict:
    return {
        "interactive_nav": {
            "interaction_requirement": "required",
            "interactions": [
                {
                    "interaction_id": "drawer_2",
                    "prerequisites": [],
                }
            ],
            "oracle_plans": [
                {
                    "plan_id": "drawer_scan",
                    "required_interaction_ids": ["drawer_2"],
                }
            ],
        }
    }


def test_drawer_scan_transient_open_counts_after_the_drawer_is_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark_metrics,
        "joint_open_fraction",
        lambda _env, _interaction: 0.0,
    )
    score = benchmark_metrics.score_interactions(
        object(),
        _episode(),
        [
            {
                "classification": "required_valid",
                "success": True,
                "resolved_interaction_ids": ["drawer_2"],
                "metadata": {
                    "transient_satisfied_interaction_ids": ["drawer_2"],
                },
            }
        ],
    )

    assert score.required_interaction_success is True
    assert score.sequence_success is True


def test_terminal_open_fraction_is_still_required_without_transient_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark_metrics,
        "joint_open_fraction",
        lambda _env, _interaction: 0.0,
    )
    score = benchmark_metrics.score_interactions(
        object(),
        _episode(),
        [
            {
                "classification": "required_valid",
                "success": True,
                "resolved_interaction_ids": ["drawer_2"],
                "metadata": {},
            }
        ],
    )

    assert score.required_interaction_success is False


def _paper_episode(requirement: str = "required") -> dict:
    return {
        "interactive_nav": {
            "interaction_requirement": requirement,
            "interactions": [
                {
                    "interaction_id": "door_1",
                    "prerequisites": [],
                }
            ],
        }
    }


def test_paper_ip_counts_effects_per_attempt_and_error_union() -> None:
    score = benchmark_metrics.paper_interaction_attempt_score(
        _paper_episode(),
        [
            {
                "classification": "required_valid",
                "success": True,
                "resolved_interaction_id": "door_1",
            },
            # Repeating a previously completed effect is one erroneous
            # attempt, even though its executor still succeeds.
            {
                "classification": "required_valid",
                "success": True,
                "resolved_interaction_id": "door_1",
            },
            # A relevant request that produces no effect is a failed attempt.
            {
                "classification": "required_valid",
                "success": False,
                "resolved_interaction_id": "door_1",
            },
            # This is both irrelevant and low-level failed, but contributes
            # only one unit to E in Total Cost.
            {
                "classification": "invalid",
                "success": False,
                "resolved_interaction_id": "wrong_object",
            },
        ],
    )

    assert score.interaction_attempt_count == 4
    assert score.valid_interaction_attempt_count == 1
    assert score.error_interaction_attempt_count == 3
    assert score.task_irrelevant_interaction_attempt_count == 1
    assert score.failed_interaction_attempt_count == 2
    assert score.repeated_interaction_attempt_count == 1
    assert score.interaction_precision_episode == 0.25


def test_paper_ip_defines_required_and_unnecessary_zero_attempt_cases() -> None:
    required = benchmark_metrics.paper_interaction_attempt_score(_paper_episode("required"), [])
    unnecessary = benchmark_metrics.paper_interaction_attempt_score(_paper_episode("unnecessary"), [])

    assert required.interaction_precision_episode == 0.0
    assert unnecessary.interaction_precision_episode == 1.0
    assert required.error_interaction_attempt_count == 0
    assert unnecessary.error_interaction_attempt_count == 0


def test_paper_failed_required_attempt_is_not_mislabelled_irrelevant() -> None:
    """A failed request for the required entity is relevant, but erroneous."""

    score = benchmark_metrics.paper_interaction_attempt_score(
        _paper_episode(),
        [
            {
                "classification": "required_valid",
                "success": False,
                # The failed executor cannot resolve an interaction ID.
            }
        ],
    )

    assert score.valid_interaction_attempt_count == 0
    assert score.failed_interaction_attempt_count == 1
    assert score.task_irrelevant_interaction_attempt_count == 0
    assert score.repeated_interaction_attempt_count == 0
    assert score.error_interaction_attempt_count == 1
    assert score.interaction_precision_episode == 0.0


def test_paper_access_blocked_ros_request_is_failed_not_irrelevant() -> None:
    """A ROS request can identify the target before approach gating rejects it."""

    score = benchmark_metrics.paper_interaction_attempt_score(
        _paper_episode(),
        [
            {
                "classification": "invalid",
                "success": False,
                "metadata": {"requested_interaction_ids": ["door_1"]},
            }
        ],
    )

    assert score.failed_interaction_attempt_count == 1
    assert score.task_irrelevant_interaction_attempt_count == 0
    assert score.error_interaction_attempt_count == 1


def test_paper_total_cost_records_all_formula_terms() -> None:
    interaction_score = benchmark_metrics.PaperInteractionAttemptScore(
        interaction_attempt_count=4,
        valid_interaction_attempt_count=1,
        error_interaction_attempt_count=3,
        task_irrelevant_interaction_attempt_count=1,
        failed_interaction_attempt_count=2,
        repeated_interaction_attempt_count=1,
        interaction_precision_episode=0.25,
    )
    config = benchmark_metrics.PaperMetricConfig(
        interaction_attempt_cost=0.5,
        error_interaction_surcharge=2.0,
        failure_penalty=7.0,
    )

    total, breakdown = benchmark_metrics.paper_episode_total_cost(
        nav_success=False,
        navigation_path_length_m=3.2,
        interaction_score=interaction_score,
        config=config,
    )

    assert total == pytest.approx(18.2)
    assert breakdown == {
        "navigation_path_length_m": 3.2,
        "interaction_attempt_cost": 2.0,
        "error_interaction_surcharge": 6.0,
        "failure_penalty": 7.0,
        "interaction_attempt_count": 4,
        "error_interaction_attempt_count": 3,
        "nav_success_indicator": 0,
        "total_cost": 18.2,
    }
