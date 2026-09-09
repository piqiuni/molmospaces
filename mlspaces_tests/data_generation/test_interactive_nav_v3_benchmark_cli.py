"""Regression guard for the public InteractiveNav V3 evaluator entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.InteractiveNav import evaluate_interactive_nav_v3 as cli
from scripts.InteractiveNav.evaluation import EvaluationConfig
from scripts.InteractiveNav.evaluation import benchmark_runner


def test_public_v3_cli_routes_to_canonical_benchmark_runner() -> None:
    """Keep formal V3 evaluation off the legacy compatibility runner."""

    assert cli.main is benchmark_runner.main
    assert EvaluationConfig is benchmark_runner.BenchmarkEvaluationConfig


def test_paper_cost_cli_parameters_are_frozen_in_manifest_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
            "--paper-cost-failure-penalty",
            "7.0",
        ],
    )

    config = benchmark_runner.parse_args()
    assert config.paper_cost_interaction_attempt == pytest.approx(0.4)
    assert config.paper_cost_error_surcharge == pytest.approx(1.2)
    assert config.paper_cost_failure_penalty == pytest.approx(7.0)

    result = benchmark_runner.run_evaluation(config)
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluation_config"]["paper_cost_interaction_attempt"] == pytest.approx(0.4)
    assert manifest["evaluation_config"]["paper_cost_error_surcharge"] == pytest.approx(1.2)
    assert manifest["evaluation_config"]["paper_cost_failure_penalty"] == pytest.approx(7.0)
    paper_metric_config = result["summary"]["paper_metric_config"]
    assert paper_metric_config["schema_version"] == "interactive_nav_v3_paper_metrics_v1"
    assert paper_metric_config["formula"] == "L_exec_m + lambda*A + mu*E + kappa*(1-S)"
    assert paper_metric_config["interaction_attempt_cost"] == pytest.approx(0.4)
    assert paper_metric_config["error_interaction_surcharge"] == pytest.approx(1.2)
    assert paper_metric_config["failure_penalty"] == pytest.approx(7.0)
    # The direct names make saved JSON ergonomic; the aliases pin the exact
    # notation used in the paper's equation.
    assert paper_metric_config["lambda_interaction_attempt_cost"] == pytest.approx(0.4)
    assert paper_metric_config["mu_error_interaction_surcharge"] == pytest.approx(1.2)
    assert paper_metric_config["kappa_failure_penalty"] == pytest.approx(7.0)

    with pytest.raises(ValueError, match="error_interaction_surcharge"):
        benchmark_runner.BenchmarkEvaluationConfig(
            benchmark=benchmark,
            output_dir=tmp_path / "invalid",
            paper_cost_interaction_attempt=1.0,
            paper_cost_error_surcharge=1.0,
            paper_cost_failure_penalty=7.0,
        ).validate()

    with pytest.raises(ValueError, match="different benchmark/evaluation signature"):
        benchmark_runner.run_evaluation(
            benchmark_runner.BenchmarkEvaluationConfig(
                benchmark=benchmark,
                output_dir=output_dir,
                resume=True,
                paper_cost_interaction_attempt=0.4,
                paper_cost_error_surcharge=1.3,
                paper_cost_failure_penalty=7.0,
            )
        )
