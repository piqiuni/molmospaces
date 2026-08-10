from __future__ import annotations

import sys
from pathlib import Path

import pytest


PACKAGE_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SCRIPTS))
MLLM_SCRIPTS = PACKAGE_SCRIPTS.parents[1] / "semantic_mllm_py_pkg" / "scripts"
if str(MLLM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MLLM_SCRIPTS))

pytest.importorskip("rospy")

from semantic_rule_decision_node import (
    aggregate_step_ready_states,
    is_completed_drawer_scan_candidate,
)


def _module(step: int, stamp: float, *, ready: bool = True, strict: bool = False):
    payload = {"ready": ready, "step_index": step, "stamp_sec": stamp}
    if strict:
        payload["causal_contract"] = "occ_room_graph_same_source"
    return payload


def test_strict_ready_rejects_different_stamps_even_when_seq_matches():
    payload = aggregate_step_ready_states(
        ("semantic_mapping", "explore_py"),
        {
            "semantic_mapping": _module(12, 10.0, strict=True),
            "explore_py": _module(12, 10.2),
        },
    )
    assert not payload["ready"]
    assert payload["source_alignment_required"]
    assert not payload["source_aligned"]
    assert payload["source_match_mode"] == "stamp"
    assert payload["missing_modules"] == ["source_alignment"]


def test_strict_ready_accepts_same_stamp_after_pipeline_resequences_headers():
    payload = aggregate_step_ready_states(
        ("semantic_mapping", "explore_py"),
        {
            "semantic_mapping": _module(75, 10.0, strict=True),
            "explore_py": _module(0, 10.0),
        },
    )
    assert payload["ready"]
    assert payload["source_aligned"]
    assert payload["source_match_mode"] == "stamp"
    assert payload["source_identity_key"] == "stamp:10.000000"
    assert payload["step_index"] == 0
    assert payload["stamp_sec"] == 10.0


def test_fast_ready_keeps_legacy_minimum_contract():
    payload = aggregate_step_ready_states(
        ("semantic_mapping", "explore_py"),
        {
            "semantic_mapping": _module(12, 10.0),
            "explore_py": _module(13, 10.2),
        },
    )
    assert payload["ready"]
    assert not payload["source_alignment_required"]
    assert payload["step_index"] == 12


def test_explicit_exact_source_falls_back_to_seq_when_stamp_unavailable():
    payload = aggregate_step_ready_states(
        ("semantic_mapping", "explore_py"),
        {
            "semantic_mapping": _module(12, 0.0),
            "explore_py": _module(13, 0.0),
        },
        require_exact_source=True,
    )
    assert not payload["ready"]
    assert payload["source_alignment_required"]
    assert payload["source_match_mode"] == "seq"


def test_completed_drawer_scan_rejects_rebuilt_candidate_for_same_target():
    completed_candidate_ids = {"interaction:drawer_1:open"}
    completed_target_ids = {"drawer_1"}
    rebuilt = {
        "candidate_id": "interaction:drawer_1:reobserve",
        "target_id": "drawer_1",
        "interaction_command": {"sequence_type": "drawer_scan"},
    }
    assert is_completed_drawer_scan_candidate(
        rebuilt, completed_candidate_ids, completed_target_ids
    )
    assert not is_completed_drawer_scan_candidate(
        {
            **rebuilt,
            "interaction_command": {"sequence_type": "drawer_open"},
        },
        completed_candidate_ids,
        completed_target_ids,
    )
