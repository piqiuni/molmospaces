"""Standalone evaluation support for the InteractiveNav V3 benchmark.

This package deliberately lives outside :mod:`molmo_spaces.evaluation`.  Its
public API and the ``evaluate_interactive_nav_v3.py`` CLI use
``benchmark_runner`` as the canonical V3 protocol implementation.  The older
``runner`` module remains importable only for compatibility with its existing
tests and experimental callers.
"""

__all__ = ["BenchmarkEvaluationConfig", "EvaluationConfig", "run_evaluation"]


def __getattr__(name: str):
    """Keep offline reporting importable without a MuJoCo runtime.

    The evaluator itself still imports ``benchmark_runner`` on demand.  This
    matters for result-only tools such as ``episode_topdown``: they only need
    NumPy/OpenCV/Matplotlib and should not require the simulator installation.
    """

    if name not in set(__all__):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .benchmark_runner import BenchmarkEvaluationConfig, run_evaluation

    values = {
        "BenchmarkEvaluationConfig": BenchmarkEvaluationConfig,
        "EvaluationConfig": BenchmarkEvaluationConfig,
        "run_evaluation": run_evaluation,
    }
    return values[name]
