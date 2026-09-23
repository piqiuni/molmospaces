from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


PACKAGE_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
MLLM_SCRIPTS = PACKAGE_SCRIPTS.parents[1] / "semantic_mllm_py_pkg" / "scripts"
for scripts in (PACKAGE_SCRIPTS, MLLM_SCRIPTS):
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

pytest.importorskip("rospy")

import semantic_rule_decision_node as decision
from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate


CANDIDATE_ID = "interaction:fridge_1:open"
TARGET_KEY = "interaction_target:fridge_1"
NAVIGATION_DETAIL = {
    "reason": "container_approach_navigation_unreachable",
    "failure_reason": "container_anchor_footprint_blocked",
    "failure_stage": "interaction_approach_navigation",
    "retryable": True,
    "action_executed": False,
    "all_container_anchors_unreachable": True,
    "observation_attempts": 0,
}


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(json.loads(message.data))


@pytest.fixture
def node(monkeypatch):
    monkeypatch.setattr(decision.rospy, "init_node", lambda *_a, **_k: None)
    monkeypatch.setattr(decision.rospy, "get_param", lambda _key, default=None: default)
    monkeypatch.setattr(decision.rospy, "Publisher", lambda *_a, **_k: Publisher())
    monkeypatch.setattr(decision.rospy, "Subscriber", lambda *_a, **_k: None)
    monkeypatch.setattr(decision.rospy, "Timer", lambda *_a, **_k: None)
    monkeypatch.setattr(decision, "load_env_file", lambda *_a, **_k: None)
    monkeypatch.setattr(decision, "apply_model_env_overrides", lambda config: config)
    return decision.SemanticRuleDecisionNode()


def test_goal_claim_waits_for_matching_evaluator_ack(node):
    node.require_goal_verification = True
    node.mission_mode = "semantic_interaction_object_goal"
    node.target_context = {"episode_id": "episode_a"}
    node._request_goal_completion({"reason": "target_goal_succeeded"})
    assert not node.goal_complete and not node.target_goal_complete
    pending = dict(node.pending_goal_claim)
    for payload in (
        {"claim_id": "old", "episode_id": "episode_a", "accepted": True},
        {"claim_id": pending["claim_id"], "episode_id": "old", "accepted": True},
    ):
        node._goal_verification_callback(SimpleNamespace(data=json.dumps(payload)))
        assert not node.goal_complete
    node._goal_verification_callback(SimpleNamespace(data=json.dumps({
        "claim_id": pending["claim_id"], "episode_id": "episode_a", "accepted": True,
    })))
    assert node.goal_complete and node.target_goal_complete
    assert node.pending_goal_claim is None


def test_rejected_goal_claim_resumes_with_fresh_candidates(node):
    node.require_goal_verification = True
    node.target_context = {"episode_id": "episode_a"}
    node.latest_candidates_payload = {"sequence": 50}
    node._request_goal_completion({"reason": "target_goal_succeeded"})
    pending = dict(node.pending_goal_claim)
    node._goal_verification_callback(SimpleNamespace(data=json.dumps({
        "claim_id": pending["claim_id"], "episode_id": "episode_a", "accepted": False,
    })))
    assert not node.goal_complete and not node.target_goal_complete
    assert node.pending_goal_claim is None
    assert node.minimum_candidate_sequence == 51
    assert node.goal_status_pub.messages[-1]["status"] == "ACTIVE"


def candidate(**metadata):
    return BehaviorCandidate(
        candidate_id=CANDIDATE_ID,
        behavior_type="INTERACT",
        source="interaction_graph",
        target_id="fridge_1",
        target_name="fridge",
        goal_xyyaw=[2.24, 0.65, 3.14],
        interaction_command={"container_kind": "fridge", "action": "open"},
        metadata=metadata,
    ).to_dict()


def snapshot(step, candidates=None, **context):
    candidates = [candidate()] if candidates is None else candidates
    return {
        "episode_id": "h5",
        "sequence": step + 1,
        "candidates": copy.deepcopy(candidates),
        "candidate_count": len(candidates),
        "robot_xy": [1.65, 3.04],
        "exploration_context": {
            "observation_step": step,
            "initial_scan_complete": True,
            "navigation_frontier_count": 0,
            "interaction_frontier_count": 1,
            "frontier_exhausted": False,
            "unresolved_interaction_target_count": 1,
            "connected_unknown_area_present": True,
            "raw_frontier_material_cluster_count": 2,
            "filtered_frontier_retryable": True,
            **context,
        },
    }


def fail(node, detail, step=426):
    node.latest_candidates_payload = snapshot(step)
    node.active_candidate_id = CANDIDATE_ID
    node.active_decision_id = f"decision_{step}"
    node.active_behavior_type = "INTERACT"
    node.active_interaction_candidate = candidate()
    node._handle_feedback({
        "status": "FAILED",
        "candidate_id": CANDIDATE_ID,
        "decision_id": node.active_decision_id,
        "detail": copy.deepcopy(detail),
    })


@pytest.mark.parametrize("detail", [
    NAVIGATION_DETAIL,
    {
        "reason": "container_m1_evidence_inconclusive_viewpoint_navigation",
        "failure_reason": "container_anchor_center_blocked",
        "failure_stage": "interaction_visual_precondition",
        "m1_evidence_inconclusive": True,
        "retryable": True,
    },
    {"reason": "drawer_m1_evidence_inconclusive_viewpoint_navigation"},
    {"all_container_anchors_unreachable": True},
    {"m1_viewpoint_navigation_inconclusive": True, "observation_attempts": 0},
    {"m1_viewpoint_navigation_inconclusive": True, "m1_capture_not_reached": True},
    {"failure_stage": "interaction_approach_navigation", "observation_attempts": 0},
])
def test_navigation_feedback_uses_short_step_cooldown_without_m1_count(node, detail, monkeypatch):
    recorded = []
    monkeypatch.setattr(node, "_record_decision_result", recorded.append)
    fail(node, detail)

    assert node.container_anchor_unreachable_until_step == {TARGET_KEY: 446}
    assert not node.cooldown_until
    assert not node.container_m1_inconclusive_counts
    assert not node.failure_counts
    assert node.interaction_failure_tracker.failure_counts == {CANDIDATE_ID: 1}
    assert not node.interaction_failure_tracker.terminal_candidate_ids
    assert not node.approach_exhausted_fingerprints
    assert node.next_decision_time == 0.0
    result = recorded[-1]["detail"]
    assert result["reason"] == "container_approach_navigation_unreachable"
    assert result["failure_stage"] == "interaction_approach_navigation"
    assert result["m1_evidence_inconclusive"] is False
    if "failure_reason" in detail:
        assert result["failure_reason"] == detail["failure_reason"]


@pytest.mark.parametrize("detail", [
    {"reason": "container_m1_evidence_inconclusive", "observation_attempts": 2},
    {"m1_viewpoint_navigation_inconclusive": True, "observation_attempts": 2},
    {"observation_attempts": 0},
    {"m1_viewpoint_navigation_inconclusive": True, "observation_attempts": "invalid"},
    {**NAVIGATION_DETAIL, "action_executed": True},
])
def test_visual_uncertainty_is_not_navigation_failure(detail):
    assert not decision.container_approach_navigation_failed(detail)


def test_genuine_m1_failure_keeps_existing_visual_retry_policy(node):
    fail(node, {
        "reason": "container_m1_evidence_inconclusive",
        "failure_stage": "interaction_visual_precondition",
        "m1_evidence_inconclusive": True,
        "retryable": True,
        "observation_attempts": 2,
    })
    assert node.container_m1_inconclusive_counts == {TARGET_KEY: 1}
    assert node.container_anchor_unreachable_until_step == {TARGET_KEY: 726}
    assert TARGET_KEY in node.cooldown_until
    assert not node.interaction_failure_tracker.terminal_candidate_ids


def test_navigation_retry_cooldown_is_step_based_and_rechecks_safety(node):
    fail(node, NAVIGATION_DETAIL)
    before = node._eligible_candidates_from_snapshot(
        snapshot(445), now=1e20, region_history={}
    )
    assert before == ([], {CANDIDATE_ID: "container_anchor_step_cooldown"})

    eligible, rejected = node._eligible_candidates_from_snapshot(
        snapshot(446), now=0.0, region_history={}
    )
    assert [item.candidate_id for item in eligible] == [CANDIDATE_ID]
    assert not rejected
    unsafe = snapshot(446, [candidate(path_reachable=False)])
    original = copy.deepcopy(unsafe)
    assert node._eligible_candidates_from_snapshot(
        unsafe, now=1e20, region_history={}
    ) == ([], {CANDIDATE_ID: "path_reachable_false"})
    assert unsafe == original


def test_navigation_retry_survives_fingerprint_and_candidate_action_change(node):
    fail(node, NAVIGATION_DETAIL)
    rebuilt = candidate()
    rebuilt["goal_xyyaw"] = [2.2, 0.9, 2.1]
    rebuilt["candidate_id"] = "interaction:fridge_1:reobserve"
    assert node._eligible_candidates_from_snapshot(
        snapshot(445, [rebuilt]), now=1e20, region_history={}
    ) == ([], {rebuilt["candidate_id"]: "container_anchor_step_cooldown"})


def test_target_preemption_ignores_ineligible_cooldown_candidate(node):
    target = {
        "candidate_id": "target:object_bowl",
        "behavior_type": "NAVIGATE",
        "source": "semantic_graph",
        "target_id": "object_bowl",
        "target_name": "bowl",
        "goal_xyyaw": [2.0, 0.0, 0.0],
        "features": {"distance_m": 2.0, "priority": 1.0},
        "metadata": {
            "target_goal": True,
            "target_visible_now": True,
            "target_reliably_observed": True,
            "target_goal_distance_m": 2.0,
        },
    }
    payload = snapshot(
        10,
        candidates=[target],
        target_context={
            "enabled": True,
            "target_name": "bowl",
            "object_labels": ["bowl"],
            "require_interaction": False,
        },
        navigation_frontier_count=1,
        interaction_frontier_count=0,
    )
    node.active_candidate_id = "frontier:old"
    node.active_decision_id = "decision_old"
    node.active_behavior_type = "EXPLORE"
    node.active_target_goal = False
    node.cooldown_until[target["candidate_id"]] = 10**12
    node._candidate_callback(SimpleNamespace(data=json.dumps(payload)))
    assert not node.preempt_pub.messages

    node.cooldown_until.clear()
    payload["sequence"] = 12
    node._candidate_callback(SimpleNamespace(data=json.dumps(payload)))
    assert node.preempt_pub.messages[-1]["replacement_candidate_id"] == target[
        "candidate_id"
    ]


def test_three_navigation_failures_use_existing_terminal_limit(node):
    for step in (426, 446, 466):
        fail(node, NAVIGATION_DETAIL, step)
    assert node.interaction_failure_tracker.terminal_candidate_ids == {CANDIDATE_ID}
    assert not node.container_m1_inconclusive_counts
    assert node._eligible_candidates_from_snapshot(
        snapshot(500), now=1e20, region_history={}
    ) == ([], {CANDIDATE_ID: "interaction_approach_terminal_unreachable"})


def test_no_step_navigation_feedback_uses_finite_wall_fallback(node, monkeypatch):
    node.latest_candidates_payload = snapshot(426)
    node.latest_candidates_payload["exploration_context"].pop("observation_step")
    node.active_candidate_id = CANDIDATE_ID
    node.active_behavior_type = "INTERACT"
    node.active_interaction_candidate = candidate()
    monkeypatch.setattr(decision.time, "monotonic", lambda: 100.0)
    node._handle_feedback({
        "status": "FAILED", "candidate_id": CANDIDATE_ID, "detail": NAVIGATION_DETAIL,
    })
    assert node.cooldown_until[TARGET_KEY] == 115.0
    assert not node.container_anchor_unreachable_until_step


def observe(tracker, step, **kwargs):
    return tracker.update(
        snapshot(step), eligible_candidate_count=0, has_active_behavior=False, **kwargs
    )


def test_idle_recovery_handles_unresolved_interaction_without_frontiers(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(decision.time, "monotonic", lambda: clock[0])
    tracker = decision.NoEligibleCandidateTracker()
    payload = snapshot(30)
    payload["exploration_context"].update(
        connected_unknown_area_present=False, raw_frontier_material_cluster_count=0,
        unresolved_interaction_target_count=1,
    )
    tracker.update(payload, eligible_candidate_count=0, has_active_behavior=False)
    clock[0] += 11
    assert tracker.recovery_candidate(payload) is not None
    clock[0] += 20
    detail = tracker.update(payload, eligible_candidate_count=0, has_active_behavior=False)
    assert detail["blocked"]
    assert detail["no_eligible_elapsed_wall_seconds"] < 60


def test_no_eligible_wait_is_bounded_by_unique_observation_steps():
    tracker = decision.NoEligibleCandidateTracker()
    for _ in range(100):
        detail = observe(tracker, 426)
        assert not detail["blocked"]
    assert detail["no_eligible_confirmations"] == 1
    assert not observe(tracker, 545)["blocked"]
    detail = observe(tracker, 546)
    assert detail["blocked"]
    assert detail["reason"] == "no_eligible_candidates_after_bounded_recovery"


def test_idle_timeout_excludes_recovery_execution(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(decision.time, "monotonic", lambda: clock[0])
    tracker = decision.NoEligibleCandidateTracker()
    payload = snapshot(30)
    payload["exploration_context"]["unresolved_interaction_target_count"] = 1
    tracker.update(payload, eligible_candidate_count=0, has_active_behavior=False)
    clock[0] = 111.0
    recovery = tracker.recovery_candidate(payload)
    assert recovery is not None
    clock[0] = 151.0
    tracker.finish_recovery(recovery.candidate_id)
    detail = tracker.update(payload, eligible_candidate_count=0, has_active_behavior=False)
    assert not detail["blocked"]
    assert detail["no_eligible_elapsed_wall_seconds"] == 11.0


def test_nonempty_pool_does_not_hide_failure_to_select_an_action(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(decision.time, "monotonic", lambda: clock[0])
    tracker = decision.NoEligibleCandidateTracker()
    payload = snapshot(30)
    detail = tracker.update(payload, eligible_candidate_count=3,
        has_active_behavior=False, has_executable_candidate=False)
    assert not detail["blocked"]
    clock[0] += 31
    detail = tracker.update(payload, eligible_candidate_count=3,
        has_active_behavior=False, has_executable_candidate=False)
    assert detail["blocked"]
    assert detail["eligible_candidate_count"] == 3
    assert detail["executable_candidate_count"] == 0
    assert tracker.update(payload, eligible_candidate_count=3,
        has_active_behavior=False, has_executable_candidate=True) == {}


def test_active_recovery_scan_keeps_idle_tracker_state(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(decision.time, "monotonic", lambda: clock[0])
    tracker = decision.NoEligibleCandidateTracker()
    payload = snapshot(30)
    payload["exploration_context"]["unresolved_interaction_target_count"] = 1
    tracker.update(payload, eligible_candidate_count=0, has_active_behavior=False)
    clock[0] = 111.0
    recovery = tracker.recovery_candidate(payload)
    assert recovery is not None
    tracker.update(payload, eligible_candidate_count=0, has_active_behavior=True)
    assert tracker.since_step == 30
    clock[0] = 121.0
    tracker.finish_recovery(recovery.candidate_id)
    assert not tracker.update(payload, eligible_candidate_count=0, has_active_behavior=False)["blocked"]


@pytest.mark.parametrize("eligible,active,scan_complete", [
    (1, False, True), (0, True, True), (0, False, False),
])
def test_no_eligible_streak_resets_for_safe_recovery_or_startup(eligible, active, scan_complete):
    tracker = decision.NoEligibleCandidateTracker()
    observe(tracker, 426)
    assert not tracker.update(
        snapshot(485, initial_scan_complete=scan_complete),
        eligible_candidate_count=eligible,
        has_active_behavior=active,
    )
    assert not observe(tracker, 486)["blocked"]
    assert tracker.since_step == 486


def test_no_eligible_tracker_handles_step_reset_and_missing_step():
    tracker = decision.NoEligibleCandidateTracker()
    observe(tracker, 426)
    assert observe(tracker, 0)["no_eligible_since_step"] == 0
    detail = tracker.update({}, eligible_candidate_count=0, has_active_behavior=False)
    assert not detail["blocked"]
    assert detail["reason"] == "no_eligible_candidates_missing_observation_step"


def decide(node, payload):
    node.latest_candidates_payload = copy.deepcopy(payload)
    node._decide_from_snapshot(copy.deepcopy(payload))


def test_node_retries_at_20_steps_without_silent_long_idle(node):
    fail(node, NAVIGATION_DETAIL)
    decide(node, snapshot(426))
    assert not node.active_candidate_id
    assert not node.goal_complete
    assert node.trace_pub.messages[-1]["no_eligible_candidate_recovery"]["reason"] == (
        "no_eligible_candidates_waiting_for_recovery"
    )
    decide(node, snapshot(446))
    assert node.active_candidate_id == CANDIDATE_ID
    assert not node.goal_complete
    assert not node.trace_pub.messages[-1]["no_eligible_candidate_recovery"]


def test_node_reports_blocked_even_with_unresolved_frontier_counters(node):
    node.container_anchor_unreachable_until_step[TARGET_KEY] = 726
    original = snapshot(426)
    for step in (426, 446, 486, 546):
        decide(node, snapshot(step))
        if node.active_behavior_type == "SCAN":
            node._handle_feedback({"candidate_id": node.active_candidate_id,
                                   "decision_id": node.active_decision_id,
                                   "status": "SUCCEEDED", "detail": {}})
    assert node.goal_complete
    status = node.goal_status_pub.messages[-1]
    assert status["status"] == "EXPLORATION_STALLED"
    assert status["detail"]["reason"] == "no_eligible_candidates_after_bounded_recovery"
    assert status["detail"]["eligibility_rejections"] == {
        CANDIDATE_ID: "container_anchor_step_cooldown",
    }
    assert status["detail"]["exploration_context"]["unresolved_interaction_target_count"] == 1
    assert node.container_anchor_unreachable_until_step[TARGET_KEY] == 726
    assert original["exploration_context"]["interaction_frontier_count"] == 1
    assert status["detail"]["recovery_scan_count"] == 2


def test_node_reports_blocked_for_material_frontiers_without_safe_viewpoints(node):
    for step in (426, 446, 486, 546):
        decide(node, snapshot(step, []))
        if node.active_behavior_type == "SCAN":
            node._handle_feedback({"candidate_id": node.active_candidate_id,
                                   "decision_id": node.active_decision_id,
                                   "status": "FAILED", "detail": {"reason": "frontier_recovery_scan_unsafe"}})
    assert node.goal_status_pub.messages[-1]["status"] == "EXPLORATION_STALLED"
    assert node.trace_pub.messages[-1]["execution_eligible_candidate_count"] == 0


def test_recovery_scan_has_two_attempts_inside_one_step_budget():
    tracker = decision.NoEligibleCandidateTracker()
    observe(tracker, 0)
    assert tracker.recovery_candidate(snapshot(0)) is None
    observe(tracker, 20)
    assert tracker.recovery_candidate(snapshot(20)).metadata["frontier_recovery_scan"]
    assert tracker.recovery_candidate(snapshot(20)) is None
    observe(tracker, 60)
    assert tracker.recovery_candidate(snapshot(60)).behavior_type == "SCAN"
    assert tracker.recovery_candidate(snapshot(60)) is None
    assert observe(tracker, 120)["blocked"]


def test_stale_costmap_feedback_does_not_consume_interaction_failure_budget(node):
    fail(node, {"reason": "navigation_costmap_not_fresh", "retryable": True})
    assert not node.interaction_failure_tracker.failure_counts
    assert not node.container_m1_inconclusive_counts
    assert node.container_anchor_unreachable_until_step == {TARGET_KEY: 446}


def test_step_only_cooldown_is_visible_to_completion_tracker(node, monkeypatch):
    node.container_anchor_unreachable_until_step[TARGET_KEY] = 446
    captured = []

    def update(payload, **_kwargs):
        captured.append(copy.deepcopy(payload))
        return False

    monkeypatch.setattr(node.completion_tracker, "update", update)
    decide(node, snapshot(426))
    assert captured[-1]["exploration_context"]["interaction_cooldown_target_count"] == 1
