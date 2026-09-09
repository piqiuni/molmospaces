from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.InteractiveNav import analyze_interactive_nav_dataset as analysis


def test_reference_path_order_matches_formal_evaluator() -> None:
    validation = {
        "initial_state_path_length_m": 9.0,
        "oracle_restored_path_length_m": 7.0,
        "path_length_m": 5.0,
        "all_open_path_length_m": 3.0,
    }

    assert analysis.gt_path_length(validation, "required") == (
        7.0,
        "oracle_restored_path_length_m",
    )
    assert analysis.gt_path_length(validation, "unnecessary") == (
        9.0,
        "initial_state_path_length_m",
    )


def test_visibility_gain_supports_container_trace_schema() -> None:
    rows = [
        {
            "visibility_trace": [
                {"visibility_fraction": 0.0, "visible_pixels": 0},
                {"visibility_fraction": 0.001, "visible_pixels": 10},
                {"visibility_fraction": 0.004, "visible_pixels": 40},
            ]
        }
    ]

    assert analysis.visibility_gain(rows) == pytest.approx((0.004, 40))


def test_visibility_gain_supports_mixed_before_after_schema() -> None:
    rows = [
        {
            "visibility_fraction_before": 0.0,
            "visibility_fraction_after": 0.002,
            "visible_pixels_before": 0,
            "visible_pixels_after": 20,
        }
    ]

    assert analysis.visibility_gain(rows) == pytest.approx((0.002, 20))


def test_scoring_manifest_filters_ineligible_rows(tmp_path: Path) -> None:
    def episode(case_id: str) -> dict[str, object]:
        return {
            "house_index": 1,
            "interactive_nav": {
                "case_id": case_id,
                "schema_version": "interactive_nav_v3",
                "interaction_domains": ["channel"],
                "interaction_requirement": "unnecessary",
                "legacy_case_type": "distractor_doors_closed",
                "target": {"category": "mug", "selected_instance": "mug_1"},
                "interactions": [],
                "oracle_plans": [{}],
                "generation_validation": {
                    "navigation_validation": {"initial_state_path_length_m": 2.0}
                },
            },
        }

    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps([episode("a"), episode("b")]))
    scoring = tmp_path / "scoring.jsonl"
    scoring.write_text(
        "\n".join(
            [
                json.dumps({
                    "episode_index": 0,
                    "case_id": "a",
                    "scoring_eligible": True,
                    "scoring_exclusion_reasons": [],
                }),
                json.dumps({
                    "episode_index": 1,
                    "case_id": "b",
                    "scoring_eligible": False,
                    "scoring_exclusion_reasons": ["runtime_exception"],
                }),
            ]
        )
        + "\n"
    )

    rows, _episodes, selection = analysis.load_rows(
        benchmark,
        scoring_manifest_path=scoring,
        eligible_only=True,
    )

    assert [row.case_id for row in rows] == ["a"]
    assert selection == {
        "source_episode_count": 2,
        "selected_episode_count": 1,
        "excluded_episode_count": 1,
        "scoring_eligible_episode_count": 1,
        "scoring_ineligible_episode_count": 1,
        "scoring_exclusion_reason_counts": {"runtime_exception": 1},
        "candidate_domain_counts": {"channel": 2},
        "selected_domain_counts": {"channel": 1},
        "scoring_ineligible_domain_counts": {"channel": 1},
    }
