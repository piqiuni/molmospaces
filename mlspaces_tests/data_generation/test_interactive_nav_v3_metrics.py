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
    assert score.required_interaction_completion_fraction == 1.0


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
    assert score.required_interaction_completion_fraction == 0.0


def test_isr_credits_one_of_two_required_effects_without_completing_the_chain(monkeypatch) -> None:
    episode = _episode()
    episode["interactive_nav"]["interactions"].insert(0, {
        "interaction_id": "door_1", "prerequisites": [],
    })
    episode["interactive_nav"]["oracle_plans"][0]["required_interaction_ids"] = [
        "door_1", "drawer_2",
    ]
    monkeypatch.setattr(
        benchmark_metrics, "joint_open_fraction",
        lambda _env, item: 1.0 if item["interaction_id"] == "door_1" else 0.0,
    )
    score = benchmark_metrics.score_interactions(object(), episode, [{
        "classification": "required_valid", "success": True,
        "resolved_interaction_id": "door_1", "metadata": {},
    }])

    assert score.required_interaction_success is False
    assert score.required_interaction_completion_fraction == 0.5
    assert score.completed_required_interaction_count == 1
    assert score.required_interaction_count == 2


def test_isr_uses_best_of_alternative_required_plans(monkeypatch) -> None:
    episode = _episode()
    episode["interactive_nav"]["interactions"].append({
        "interaction_id": "fridge_1", "prerequisites": [],
    })
    episode["interactive_nav"]["oracle_plans"].append({
        "plan_id": "alternative", "required_interaction_ids": ["fridge_1"],
    })
    monkeypatch.setattr(
        benchmark_metrics, "joint_open_fraction",
        lambda _env, item: 1.0 if item["interaction_id"] == "fridge_1" else 0.0,
    )
    score = benchmark_metrics.score_interactions(object(), episode, [{
        "classification": "required_valid", "success": True,
        "resolved_interaction_id": "fridge_1", "metadata": {},
    }])

    assert score.required_interaction_success is True
    assert score.required_interaction_completion_fraction == 1.0
    assert score.required_interaction_count == 1


def _paper_episode(requirement: str = "required") -> dict:
    return {
        "interactive_nav": {
            "interaction_requirement": requirement,
            "interactions": [
                {
                    "interaction_id": "door_1",
                    "type": "channel_hinged_door",
                    "object_category": "Door",
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
                "metadata": {"physical_effect_achieved": True},
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
                "metadata": {
                    "resolved_object_category": "fridge",
                    "resolved_object_domain": "container",
                },
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


def test_paper_ip_credits_successful_exploration_of_target_class_without_oracle_id() -> None:
    score = benchmark_metrics.paper_interaction_attempt_score(
        _paper_episode(),
        [
            {"success": True, "resolved_object_name": "other_door", "metadata": {
                "physical_effect_achieved": True,
                "resolved_object_category": "door", "resolved_object_domain": "channel",
            }},
            {"success": True, "resolved_object_name": "other_door", "metadata": {
                "resolved_object_category": "door", "resolved_object_domain": "channel",
            }},
            {"success": True, "resolved_object_name": "other_fridge", "metadata": {
                "physical_effect_achieved": True,
                "resolved_object_category": "Fridge", "resolved_object_domain": "container",
            }},
            {"success": False, "resolved_object_name": "door_1", "metadata": {
                "requested_interaction_ids": ["door_1"],
            }},
        ],
    )

    assert score.valid_interaction_attempt_count == 1
    assert score.non_target_class_interaction_attempt_count == 1
    assert score.failed_interaction_attempt_count == 1
    assert score.repeated_interaction_attempt_count == 1
    assert score.error_interaction_attempt_count == 2
    assert score.interaction_precision_episode == pytest.approx(0.25)


def test_paper_cost_charges_other_class_exploration_only_once() -> None:
    score = benchmark_metrics.paper_interaction_attempt_score(
        _paper_episode(),
        [{"success": True, "resolved_object_name": "fridge_a", "metadata": {
            "physical_effect_achieved": True,
            "resolved_object_category": "Fridge", "resolved_object_domain": "container",
        }}],
    )
    cost, breakdown = benchmark_metrics.paper_episode_total_cost(
        nav_success=True,
        navigation_path_length_m=2.0,
        interaction_score=score,
        config=benchmark_metrics.PaperMetricConfig(),
    )
    assert score.interaction_precision_episode == 0.0
    assert score.error_interaction_attempt_count == 0
    assert breakdown["error_interaction_surcharge"] == 0.0
    assert cost == pytest.approx(2.3 / 30)


def test_paper_ip_requires_a_new_physical_effect_for_another_target_class_instance() -> None:
    score = benchmark_metrics.paper_interaction_attempt_score(
        _paper_episode(),
        [{"success": True, "resolved_object_name": "other_door", "metadata": {
            "resolved_object_category": "Door", "resolved_object_domain": "channel",
            "physical_state_changed": False,
        }}],
    )
    assert score.valid_interaction_attempt_count == 0
    assert score.interaction_precision_episode == 0.0


def test_paper_ip_mixed_uses_the_union_of_target_classes() -> None:
    episode = _paper_episode()
    episode["interactive_nav"]["interactions"].append({
        "interaction_id": "fridge_1", "type": "container_hinged_door",
        "object_category": "Fridge",
    })
    score = benchmark_metrics.paper_interaction_attempt_score(
        episode,
        [
            {"success": True, "resolved_object_name": "other_door", "metadata": {
                "resolved_object_category": "Door", "resolved_object_domain": "channel",
                "physical_effect_achieved": True,
            }},
            {"success": True, "resolved_object_name": "other_fridge", "metadata": {
                "resolved_object_category": "fridge", "resolved_object_domain": "container",
                "physical_effect_achieved": True,
            }},
        ],
    )
    assert score.valid_interaction_attempt_count == 2
    assert score.interaction_precision_episode == 1.0
    assert score.error_interaction_attempt_count == 0


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


@pytest.mark.parametrize("success,budget,expected", [(False, 20, 1), (True, 20, 0.56), (True, 10, 1), (True, 11.2, 1)])
def test_paper_total_cost_records_all_formula_terms(success, budget, expected) -> None:
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
        cost_budget=budget,
    )

    total, breakdown = benchmark_metrics.paper_episode_total_cost(
        nav_success=success,
        navigation_path_length_m=3.2,
        interaction_score=interaction_score,
        config=config,
    )

    assert total == pytest.approx(expected)
    assert breakdown.pop("total_cost") == pytest.approx(expected)
    assert breakdown == {
        "navigation_path_length_m": 3.2,
        "interaction_attempt_cost": 2.0,
        "error_interaction_surcharge": 6.0,
        "operation_cost": 11.2,
        "cost_budget": float(budget),
        "cost_budget_exceeded": 11.2 > budget,
        "success_within_cost_budget": success and 11.2 <= budget,
        "interaction_attempt_count": 4,
        "error_interaction_attempt_count": 3,
        "nav_success_indicator": int(success),
    }


@pytest.mark.parametrize("transient", [False, True])
def test_isr_credits_partial_effects_of_failed_macro(monkeypatch, transient) -> None:
    episode = _episode()
    episode["interactive_nav"]["interactions"].append({"interaction_id": "drawer_3", "prerequisites": []})
    episode["interactive_nav"]["oracle_plans"][0]["required_interaction_ids"].append("drawer_3")
    monkeypatch.setattr(benchmark_metrics, "joint_open_fraction", lambda env, row: float(row["interaction_id"] == "drawer_2" and not transient))
    metadata = {"effect_achieved_interaction_ids": ["drawer_2"]}
    if transient:
        metadata["transient_satisfied_interaction_ids"] = ["drawer_2"]
    score = benchmark_metrics.score_interactions(object(), episode, [{
        "classification": "required_valid", "success": False, "metadata": metadata,
    }])
    assert score.required_interaction_completion_fraction == 0.5
    assert not score.required_interaction_success
    # Merely requesting a joint, including a legacy singular resolution on a
    # failed attempt, does not prove that the physical effect occurred.
    score = benchmark_metrics.score_interactions(object(), episode, [{
        "classification": "required_valid", "success": False,
        "resolved_interaction_id": "drawer_2",
        "metadata": {"requested_interaction_ids": ["drawer_2"]},
    }])
    assert score.required_interaction_completion_fraction == 0


@pytest.mark.parametrize("before,after,success,credit", [
    (0.0, 1.0, True, 1), (0.2, 0.8, True, 1),
    (0.1, 0.4, True, 0), (0.9, 1.0, True, 0), (1.0, 1.0, True, 0),
    (0.0, 1.0, False, 1),
])
def test_ip_counts_current_threshold_crossing_instead_of_lifetime_novelty(before, after, success, credit) -> None:
    attempt = {"success": success, "resolved_interaction_id": "door_1",
               "joint_fraction_before": before, "joint_fraction_after": after}
    score = benchmark_metrics.paper_interaction_attempt_score(_paper_episode(), [attempt, attempt])
    assert score.valid_interaction_attempt_count == 2 * credit
    assert score.interaction_precision_episode == credit
    assert score.error_interaction_attempt_count == 2 * (1 - credit)


def test_ip_counts_transient_macro_effect_once_even_when_macro_fails() -> None:
    attempt = {"success": False, "resolved_interaction_ids": ["door_1"],
               "metadata": {"physical_effect_achieved": True}}
    score = benchmark_metrics.paper_interaction_attempt_score(_paper_episode(), [attempt])
    assert score.interaction_precision_episode == 1
    assert score.error_interaction_attempt_count == 0


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_normalized_cost_rejects_invalid_budget(budget) -> None:
    with pytest.raises(ValueError, match="cost_budget"):
        benchmark_metrics.PaperMetricConfig(cost_budget=budget).validate()


def test_zero_attempt_cost_distinguishes_initial_success_and_early_failure() -> None:
    score = benchmark_metrics.paper_interaction_attempt_score(_paper_episode(), [])
    for success, expected in ((True, 0.0), (False, 1.0)):
        cost, details = benchmark_metrics.paper_episode_total_cost(
            nav_success=success, navigation_path_length_m=0,
            interaction_score=score, config=benchmark_metrics.PaperMetricConfig(),
        )
        assert cost == expected
        assert details["operation_cost"] == 0
