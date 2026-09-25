"""Opt-in module ablations; importing this package does not alter Full."""

from pathlib import Path
import sys

VARIANTS = ("full", "no_interaction_graph", "no_task_decision", "no_outcome_update")
BASELINE_COMMIT = "154aa0226e7ce42bcbc8b3efe31bfd1990ab5a36"
DESIGN_REVISION = 3


def add_source_paths(repo: Path) -> None:
    for package in ("semantic_decision_py_pkg", "semantic_mapping_py_pkg", "semantic_mllm_py_pkg"):
        path = str(repo / "Interactive-Nav-SG-nav/src" / package / "scripts")
        if path not in sys.path:
            sys.path.insert(0, path)
