"""Standalone evaluation support for the InteractiveNav V3 benchmark.

This package deliberately lives outside :mod:`molmo_spaces.evaluation`.  Its
public API and the ``evaluate_interactive_nav_v3.py`` CLI use
``benchmark_runner`` as the canonical V3 protocol implementation.  The older
``runner`` module remains importable only for compatibility with its existing
tests and experimental callers.
"""

from .benchmark_runner import BenchmarkEvaluationConfig, run_evaluation

# Keep the concise historical name for callers importing the package-level
# API, while pointing it at the canonical benchmark protocol.
EvaluationConfig = BenchmarkEvaluationConfig

__all__ = ["BenchmarkEvaluationConfig", "EvaluationConfig", "run_evaluation"]
