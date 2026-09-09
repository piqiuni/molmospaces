"""Offline contract tests for the mixed benchmark scheduler."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "InteractiveNav"
    / "run_interactive_nav_benchmark_eval.py"
)
SPEC = importlib.util.spec_from_file_location("interactive_nav_mixed_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_selector_round_robins_three_domains() -> None:
    selected = MODULE.select_domain_episodes(
        {
            "channel": [{"id": index} for index in range(5)],
            "container": [{"id": index} for index in range(5)],
            "mixed": [{"id": index} for index in range(5)],
        },
        max_episodes=7,
    )

    assert [(item.domain, item.source_index) for item in selected] == [
        ("channel", 0),
        ("container", 0),
        ("mixed", 0),
        ("channel", 1),
        ("container", 1),
        ("mixed", 1),
        ("channel", 2),
    ]


def test_selector_caps_each_domain_and_preserves_local_indices() -> None:
    selected = MODULE.select_domain_episodes(
        {
            "channel": [{"id": index} for index in range(2)],
            "container": [{"id": index} for index in range(4)],
            "mixed": [{"id": index} for index in range(3)],
        },
        episodes_per_domain=3,
    )

    assert [(item.domain, item.source_index) for item in selected] == [
        ("channel", 0),
        ("container", 0),
        ("mixed", 0),
        ("channel", 1),
        ("container", 1),
        ("mixed", 1),
        ("container", 2),
        ("mixed", 2),
    ]


def test_progress_formal_sr_uses_result_success_and_eligibility() -> None:
    rows = [
        {"success": True, "nav_success": False, "scoring_eligible": True},
        {"success": False, "nav_success": True, "scoring_eligible": True},
        {"success": True, "nav_success": True, "scoring_eligible": False},
    ]

    assert MODULE._episode_sr(rows) == 0.5
    assert MODULE._episode_nav_sr(rows) == 0.5
