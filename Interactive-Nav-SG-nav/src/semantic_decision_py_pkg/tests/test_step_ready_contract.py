from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PACKAGE_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SCRIPTS))
MLLM_SCRIPTS = PACKAGE_SCRIPTS.parents[1] / "semantic_mllm_py_pkg" / "scripts"
if str(MLLM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MLLM_SCRIPTS))

pytest.importorskip("rospy")

from semantic_rule_decision_node import (
    SemanticRuleDecisionNode,
    aggregate_step_ready_states,
    container_anchor_step_cooldown_active,
    is_completed_drawer_scan_candidate,
    successful_drawer_scan_feedback,
)


def test_container_anchor_step_cooldown_is_deterministic() -> None:
    deadlines = {"interaction_target:fridge": 420}

    assert container_anchor_step_cooldown_active(
        "interaction_target:fridge", 119, deadlines
    )
    assert container_anchor_step_cooldown_active(
        "interaction_target:fridge", 419, deadlines
    )
    assert not container_anchor_step_cooldown_active(
        "interaction_target:fridge", 420, deadlines
    )
    assert not container_anchor_step_cooldown_active("", 119, deadlines)


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


def test_completed_drawer_scan_recognizes_pre_m1_drawer_candidate_shape():
    candidate = {
        "candidate_id": "interaction:drawer_1:open",
        "target_id": "drawer_1",
        "interaction_command": {"container_kind": "drawer", "action": "open"},
    }
    assert is_completed_drawer_scan_candidate(
        candidate, set(), {"drawer_1"}
    )


def test_drawer_scan_success_uses_executor_detail_over_stale_candidate_shape():
    active_candidate = {
        "interaction_command": {"container_kind": "drawer", "action": "open"}
    }
    assert successful_drawer_scan_feedback(
        {"status": "SUCCEEDED", "detail": {"sequence_type": "drawer_scan"}},
        active_candidate,
    )
    assert not successful_drawer_scan_feedback(
        {"status": "SUCCEEDED", "detail": {"sequence_type": "drawer_open"}},
        active_candidate,
    )


def test_feedback_tombstones_real_drawer_result_before_candidate_rebuild():
    candidate_id = "interaction:drawer_1:open"
    target_id = "drawer_1"
    node = SimpleNamespace(
        active_decision_id="decision_1",
        active_behavior_type="INTERACT",
        active_interaction_candidate={
            "candidate_id": candidate_id,
            "target_id": target_id,
            # This is the pre-M1 candidate shape retained by the decision node.
            "interaction_command": {"container_kind": "drawer", "action": "open"},
        },
        ablation=SimpleNamespace(module3="mllm_skill_verified"),
        direct_atomic_outcome_belief_enabled=False,
        interaction_failure_tracker=SimpleNamespace(note_feedback=lambda **_: {}),
        completion_tracker=SimpleNamespace(note_feedback=lambda _: None),
        terminal_no_plan_exit_tracker=SimpleNamespace(
            note_feedback=lambda *_args, **_kwargs: None
        ),
        latest_candidates_payload={"sequence": 7, "candidates": [], "robot_xy": []},
        failure_counts={},
        cooldown_until={},
        success_cooldown_s=0.0,
        failure_cooldown_schedule_s=(),
        interaction_target_failure_cooldown_s=0.0,
        failure_retry_delay_s=0.0,
        next_decision_time=0.0,
        pending_post_interaction_traversal={},
        terminal_post_interaction_traversal_ids=set(),
        completed_post_interaction_traversal_event_keys=set(),
        completed_drawer_scan_candidate_ids=set(),
        completed_drawer_scan_target_ids=set(),
        target_mission=SimpleNamespace(matches_target_interaction=lambda **_: False),
        target_context={},
        active_target_goal=False,
        target_goal_complete=False,
        goal_complete=False,
        mission_mode="semantic_exploration",
        portal_traversal_distance_m=1.0,
        minimum_candidate_sequence=0,
        active_candidate_id=candidate_id,
        preempt_requested_for_decision_id="",
        _record_decision_result=lambda _: None,
        _observation_step=lambda _: 0,
        _interaction_target_id=lambda _: target_id,
        _post_interaction_traversal_metadata=lambda _: {},
        _publish_inactive_selection=lambda _: None,
    )
    feedback = {
        "status": "SUCCEEDED",
        "behavior_type": "INTERACT",
        "candidate_id": candidate_id,
        "decision_id": "decision_1",
        "target_id": target_id,
        # The executor result is the only place sequence_type is guaranteed.
        "detail": {"sequence_type": "drawer_scan", "node_id": target_id},
    }

    SemanticRuleDecisionNode._handle_feedback(node, feedback)

    assert node.completed_drawer_scan_candidate_ids == {candidate_id}
    assert node.completed_drawer_scan_target_ids == {target_id}
    assert is_completed_drawer_scan_candidate(
        {
            "candidate_id": candidate_id,
            "target_id": target_id,
            "interaction_command": {"container_kind": "drawer", "action": "open"},
        },
        node.completed_drawer_scan_candidate_ids,
        node.completed_drawer_scan_target_ids,
    )
