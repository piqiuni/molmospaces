from __future__ import annotations

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
