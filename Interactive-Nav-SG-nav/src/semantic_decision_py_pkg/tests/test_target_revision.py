"""Offline tests for instruction changes across candidate generation and M2."""

import copy
import json
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("rospy")
from std_msgs.msg import String
from semantic_candidate_node import SemanticCandidateNode
from semantic_rule_decision_node import SemanticRuleDecisionNode, candidate_mission_token
from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate


def test_candidate_publish_rejects_goal_changed_during_generation_and_aba():
    node = object.__new__(SemanticCandidateNode)
    node._target_lock = threading.Lock()
    node.target_context = {"enabled": True, "target_name": "fridge"}
    node.target_revision = 0
    published = []
    node.publisher = SimpleNamespace(publish=lambda msg: published.append(json.loads(msg.data)))
    revision, context = node._target_snapshot()
    payload = {"target_revision": revision, "target_context": context, "candidates": []}
    node._target_callback(String(data=json.dumps({"enabled": True, "target_name": "door"})))
    assert not node._publish_target_snapshot(payload, revision)
    assert context["target_name"] == "fridge"
    node._target_callback(String(data=json.dumps(context)))
    assert not node._publish_target_snapshot(payload, revision)
    new_revision, new_context = node._target_snapshot()
    assert new_revision == 2
    assert node._publish_target_snapshot({"target_revision": new_revision,
                                          "target_context": new_context}, new_revision)
    assert len(published) == 1
    node._target_callback(String(data=json.dumps(context)))
    assert node.target_revision == 2  # An identical retransmission is not a new task.


@pytest.mark.parametrize("change", ["target", "revision", "episode"])
def test_late_m2_answer_is_discarded_when_mission_changes_during_model_call(change):
    node = object.__new__(SemanticRuleDecisionNode)
    node.state_lock = threading.RLock()
    snapshot = {"episode_id": "run-a", "sequence": 10, "graph_revision": 20,
                "target_revision": 1, "target_context": {"enabled": True, "target_name": "fridge"}}
    node.latest_candidates_payload = copy.deepcopy(snapshot)
    node.target_context = snapshot["target_context"]
    node.next_decision_time = 100.0
    node.pending_post_interaction_traversal = {}
    node.ablation = SimpleNamespace(module3="external_mllm_verified")
    node.direct_atomic_outcome_belief_enabled = False
    node._completion_snapshot_without_terminal_interactions = lambda value: (value, [])
    node.cooldown_until = {}
    node.minimum_candidate_sequence = 0
    node.mission_mode = "semantic_interaction_object_goal"
    node.target_goal_complete = False
    node.active_candidate_id = ""
    node.priority_target_candidate_id = ""
    node.completion_tracker = SimpleNamespace(update=lambda *a, **k: False, terminal_stalled=False)
    node._history_context = lambda value: ([], {})
    node._region_history_context = lambda value: {}
    candidate = BehaviorCandidate(candidate_id="same-anchor", behavior_type="NAVIGATE",
                                  source="test", target_id="object-a", target_name="object",
                                  goal_xyyaw=[1.0, 2.0, 0.0])
    node._eligible_candidates_from_snapshot = lambda *a, **k: ([candidate], {})
    node.target_mission = SimpleNamespace(priority_target_candidate=lambda values: None)
    node._entered_rooms_snapshot = lambda: []
    curation = SimpleNamespace(candidates=[candidate], quality_by_id={}, quality_terms_by_id={},
                               decision_hint_by_id={}, entered_room_ids=[], mandatory_ids=[])
    node.candidate_curator = SimpleNamespace(config=SimpleNamespace(region_size_m=1.0),
                                             curate=lambda *a, **k: curation)
    node.policy_backend = "model"
    node.model_circuit_breaker = SimpleNamespace(allow_request=lambda now: True,
                                                 record_success=lambda: None)
    model_inputs, selected, traces = [], [], []
    node.selected_pub = SimpleNamespace(publish=selected.append)
    node.trace_pub = SimpleNamespace(publish=lambda msg: traces.append(json.loads(msg.data)))

    def model_select(*args, **kwargs):
        model_inputs.append(kwargs["target_context"])
        newer = copy.deepcopy(snapshot)
        if change == "target":
            newer["target_context"]["target_name"] = "door"
        elif change == "revision":
            newer["target_revision"] = 3  # A -> B -> A while M2 was running.
        else:
            newer["episode_id"] = "run-b"
        # Keep candidate sequence and geometry unchanged to isolate mission identity.
        with node.state_lock:
            node.latest_candidates_payload = newer
        return candidate

    node.model_policy = SimpleNamespace(config=SimpleNamespace(selection_granularity="candidate"),
                                        select=model_select, last_result_source="model")
    node._decide_from_snapshot(copy.deepcopy(snapshot))
    assert model_inputs == [snapshot["target_context"]]
    assert selected == []
    assert traces[-1]["phase"] == "stale_mission_discarded"
    assert node.next_decision_time == 0.0


def test_mission_identity_does_not_change_on_camera_or_graph_heartbeat():
    before = {"episode_id": "run", "sequence": 1, "graph_revision": 2,
              "target_revision": 3, "target_context": {"enabled": True, "target_name": "fridge"}}
    after = {**before, "sequence": 100, "graph_revision": 90}
    assert candidate_mission_token(before) == candidate_mission_token(after)


@pytest.mark.parametrize("behavior", ["NAVIGATE", "INTERACT"])
@pytest.mark.parametrize("new_episode", [False, True])
def test_old_goal_arrival_releases_executor_without_completing_new_task(behavior, new_episode):
    node = object.__new__(SemanticRuleDecisionNode)
    old = {"episode_id": "run", "target_revision": 1,
           "target_context": {"enabled": True, "target_name": "fridge"}}
    node._active_mission_token = candidate_mission_token(old)
    node.latest_candidates_payload = {**old, "target_revision": 2,
                                       "episode_id": "run-new" if new_episode else "run",
                                       "target_context": {"enabled": True, "target_name": "door"}}
    node.active_candidate_id = "fridge-approach"
    node.active_decision_id = "decision_1"
    node.active_behavior_type = behavior
    node.active_interaction_candidate = {}
    node.active_target_goal = True
    node.target_goal_complete = False
    node.goal_complete = False
    inactive = []
    node._publish_inactive_selection = inactive.append
    feedback = {"candidate_id": "fridge-approach", "decision_id": "decision_1",
                "episode_id": "run", "status": "SUCCEEDED"}
    for invalid in ({**feedback, "episode_id": "foreign"},
                    {**feedback, "decision_id": ""},
                    {**feedback, "decision_id": "other-decision"}):
        node._handle_feedback(invalid)
        assert node.active_decision_id == "decision_1"
        assert not inactive
    node._handle_feedback(feedback)
    node._handle_feedback(feedback)
    assert node.active_decision_id == ""
    assert node.target_goal_complete is False and node.goal_complete is False
    assert inactive[0]["detail"]["mission_result_ignored"] == "target_or_episode_changed"
    assert len(inactive) == 1


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("change", ["revision", "episode", "both"])
def test_new_goal_clears_old_continuation_and_completion_counters(active, change):
    node = object.__new__(SemanticRuleDecisionNode)
    node.state_lock = threading.RLock()
    node.target_context = {"enabled": True, "target_name": "fridge"}
    node.latest_candidates_payload = {"episode_id": "run", "target_revision": 1,
                                       "target_context": node.target_context}
    node.pending_post_interaction_traversal = {"candidate_id": "old-door-traverse"}
    resets = []
    node.target_mission = SimpleNamespace(reset=lambda: resets.append("mission"),
                                          priority_target_candidate=lambda values: None)
    node.post_interaction_refresh_gate = SimpleNamespace(clear=lambda: resets.append("refresh"))
    node.completion_tracker = SimpleNamespace(reset=lambda: resets.append("completion"))
    node.terminal_no_plan_exit_tracker = SimpleNamespace(reset=lambda: resets.append("no_plan"))
    for name in (
        "terminal_post_interaction_traversal_ids", "completed_post_interaction_traversal_event_keys",
        "completed_drawer_scan_candidate_ids", "completed_drawer_scan_target_ids",
        "entered_room_ids",
    ):
        setattr(node, name, set())
    for name in (
        "interaction_outcome_beliefs", "cooldown_until", "failure_counts",
        "container_m1_inconclusive_counts", "container_anchor_unreachable_until_step",
        "approach_exhausted_fingerprints", "decision_history", "group_history", "region_history",
    ):
        setattr(node, name, {})
    node.interaction_failure_tracker = SimpleNamespace(reset=lambda: None)
    node.model_circuit_breaker = SimpleNamespace(consecutive_timeout_limit=3, cooldown_s=10.0)
    node._update_entered_rooms = lambda payload: None
    node.active_candidate_id = "old-anchor" if active else ""
    node.active_decision_id = "old-decision" if active else ""
    node.active_behavior_type = "NAVIGATE"
    node.active_interaction_candidate = {}
    node.active_target_goal = True
    node._active_mission_token = candidate_mission_token(node.latest_candidates_payload) if active else None
    node.preempt_requested_for_decision_id = ""
    requests = []
    node.preempt_pub = SimpleNamespace(publish=lambda msg: requests.append(json.loads(msg.data)))
    node._reproject_delegated_pending_traversal_locked = lambda: None
    node._publish_goal_status = lambda status: None
    newer = {**node.latest_candidates_payload,
             "target_revision": 3 if change != "episode" else 1,
             "episode_id": "run-new" if change != "revision" else "run"}
    node._candidate_callback(String(data=json.dumps(newer)))
    assert node.pending_post_interaction_traversal == {}
    if change == "revision":
        assert resets == ["mission", "refresh", "completion", "no_plan"]
    else:
        assert resets.count("mission") == (2 if change == "both" else 1)
        assert resets.count("completion") == (2 if change == "both" else 1)
    assert node.latest_candidates_payload == newer
    node._candidate_callback(String(data=json.dumps(newer)))
    assert len(requests) == int(active)
    if active:
        assert requests[0]["reason"] == "mission_changed"
        assert requests[0]["decision_id"] == "old-decision"
        # Do not release the slot until the executor's stop feedback arrives.
        assert node.active_decision_id == "old-decision"
