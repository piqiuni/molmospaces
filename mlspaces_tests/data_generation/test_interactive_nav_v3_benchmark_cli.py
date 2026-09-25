"""Regression guard for the public InteractiveNav V3 evaluator entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.InteractiveNav import evaluate_interactive_nav_v3 as cli
from scripts.InteractiveNav.evaluation import EvaluationConfig
from scripts.InteractiveNav.evaluation import benchmark_runner


def test_v3_runner_keeps_the_expanded_m1_budget_isolated() -> None:
    """M1 JSON headroom must not silently retune M2, M3, or room inference."""

    runner = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "InteractiveNav"
        / "run_interactive_nav_v3_ros_eval_test.zsh"
    )
    source = runner.read_text(encoding="utf-8")
    assert 'source "${DEFAULT_EVAL_CONFIG}"' in source
    source += (runner.parent / "configs/evaluation/benchmark_eval.conf").read_text(encoding="utf-8")

    assert "SEMANTIC_ATTRIBUTE_MAX_OUTPUT_TOKENS=${SEMANTIC_ATTRIBUTE_MAX_OUTPUT_TOKENS:-384}" in source
    assert 'semantic_attribute_max_output_tokens:="${SEMANTIC_ATTRIBUTE_MAX_OUTPUT_TOKENS}"' in source
    assert "m1_attribute_max_output_tokens=${SEMANTIC_ATTRIBUTE_MAX_OUTPUT_TOKENS}" in source
    assert "object_goal_v3_full_mllm.yaml and room MLLM keeps its own mapping config cap" in source


def test_v3_runner_bounds_observation_turns_and_gives_m1_worker_pool_headroom() -> None:
    runner = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "InteractiveNav"
        / "run_interactive_nav_v3_ros_eval_test.zsh"
    )
    source = runner.read_text(encoding="utf-8")
    assert 'source "${DEFAULT_EVAL_CONFIG}"' in source
    source += (runner.parent / "configs/evaluation/benchmark_eval.conf").read_text(encoding="utf-8")

    assert "SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S=${SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S:-30.0}" in source
    assert 'semantic_attribute_request_timeout_s:="${SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S}"' in source
    assert "ROS_ACTION_TIMEOUT_S=${ROS_ACTION_TIMEOUT_S:-0.4}" in source
    assert "ROS_STEP_READY_BARRIER_ENABLED=${ROS_STEP_READY_BARRIER_ENABLED:-true}" in source
    assert '--ros-step-ready-topic "${ROS_STEP_READY_TOPIC}"' in source
    assert "EVAL_ARGS+=(--ros-step-ready-barrier-enabled)" in source
    assert "ROS_OBSERVATION_TURN_MULTIPLIER=${ROS_OBSERVATION_TURN_MULTIPLIER:-1.5}" in source
    assert '--ros-observation-turn-multiplier "${ROS_OBSERVATION_TURN_MULTIPLIER}"' in source


def test_v3_runner_matches_zero_padded_episode_result_directory() -> None:
    """The evaluator writes ``0000_<case>`` directories, not ``0_<case>``."""

    runner = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "InteractiveNav"
        / "run_interactive_nav_v3_ros_eval_test.zsh"
    )
    source = runner.read_text(encoding="utf-8")
    assert 'source "${DEFAULT_EVAL_CONFIG}"' in source
    source += (runner.parent / "configs/evaluation/benchmark_eval.conf").read_text(encoding="utf-8")

    assert 'EPISODE_INDEX_PADDED=$(printf \'%04d\' "${EPISODE_INDEX}")' in source
    assert '${EPISODE_INDEX_PADDED}_*/episode_result.json' in source
    assert '${EPISODE_INDEX}_*/episode_result.json' not in source


def test_public_v3_cli_routes_to_canonical_benchmark_runner() -> None:
    """Keep formal V3 evaluation off the legacy compatibility runner."""

    assert cli.main is benchmark_runner.main
    assert EvaluationConfig is benchmark_runner.BenchmarkEvaluationConfig


@pytest.mark.parametrize(
    ("resume_error_surcharge", "resume_cost_budget"), [(1.3, 7.0), (1.2, 8.0)]
)
def test_paper_cost_cli_parameters_are_frozen_in_manifest_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resume_error_surcharge: float,
    resume_cost_budget: float,
) -> None:
    """Paper cost weights must be explicit, validated, and resume-locked."""

    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({"episodes": []}), encoding="utf-8")
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_interactive_nav_v3.py",
            "--benchmark",
            str(benchmark),
            "--output-dir",
            str(output_dir),
            "--paper-cost-interaction-attempt",
            "0.4",
            "--paper-cost-error-surcharge",
            "1.2",
            "--paper-cost-budget",
            "7.0",
        ],
    )

    config = benchmark_runner.parse_args()
    assert config.paper_cost_interaction_attempt == pytest.approx(0.4)
    assert config.paper_cost_error_surcharge == pytest.approx(1.2)
    assert config.paper_cost_budget == pytest.approx(7.0)

    result = benchmark_runner.run_evaluation(config)
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluation_config"]["paper_cost_interaction_attempt"] == pytest.approx(0.4)
    assert manifest["evaluation_config"]["paper_cost_error_surcharge"] == pytest.approx(1.2)
    assert manifest["evaluation_config"]["paper_cost_budget"] == pytest.approx(7.0)
    paper_metric_config = result["summary"]["paper_metric_config"]
    assert paper_metric_config["schema_version"] == "interactive_nav_v3_paper_metrics_v4"
    assert paper_metric_config["interaction_success_definition"] == "episode_mean_of_best_required_plan_effect_completion_fraction"
    assert paper_metric_config["formula"] == "min((L_exec_m + lambda*A + mu*E)/B, 1) if S else 1"
    assert paper_metric_config["error_definition"] == "failed_or_effect_free_repeated_attempts"
    assert paper_metric_config["interaction_attempt_cost"] == pytest.approx(0.4)
    assert paper_metric_config["error_interaction_surcharge"] == pytest.approx(1.2)
    assert paper_metric_config["cost_budget"] == pytest.approx(7.0)
    # The direct names make saved JSON ergonomic; the aliases pin the exact
    # notation used in the paper's equation.
    assert paper_metric_config["lambda_interaction_attempt_cost"] == pytest.approx(0.4)
    assert paper_metric_config["mu_error_interaction_surcharge"] == pytest.approx(1.2)
    assert paper_metric_config["B_cost_budget"] == pytest.approx(7.0)

    with pytest.raises(ValueError, match="error_interaction_surcharge"):
        benchmark_runner.BenchmarkEvaluationConfig(
            benchmark=benchmark,
            output_dir=tmp_path / "invalid",
            paper_cost_interaction_attempt=1.0,
            paper_cost_error_surcharge=1.0,
            paper_cost_budget=7.0,
        ).validate()

    with pytest.raises(ValueError, match="different benchmark/evaluation signature"):
        benchmark_runner.run_evaluation(
            benchmark_runner.BenchmarkEvaluationConfig(
                benchmark=benchmark,
                output_dir=output_dir,
                resume=True,
                paper_cost_interaction_attempt=0.4,
                paper_cost_error_surcharge=resume_error_surcharge,
                paper_cost_budget=resume_cost_budget,
            )
        )
