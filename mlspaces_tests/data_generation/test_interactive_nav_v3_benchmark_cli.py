"""Regression guard for the public InteractiveNav V3 evaluator entry point."""

from __future__ import annotations

from scripts.InteractiveNav import evaluate_interactive_nav_v3 as cli
from scripts.InteractiveNav.evaluation import EvaluationConfig
from scripts.InteractiveNav.evaluation import benchmark_runner


def test_public_v3_cli_routes_to_canonical_benchmark_runner() -> None:
    """Keep formal V3 evaluation off the legacy compatibility runner."""

    assert cli.main is benchmark_runner.main
    assert EvaluationConfig is benchmark_runner.BenchmarkEvaluationConfig
