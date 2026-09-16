"""Real decision-node lifecycle with mocked ROS transport; no master or model."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("rospy")
import semantic_rule_decision_node as decision
from std_msgs.msg import String

from ablations.nodes import decision_node_class, mapping_node_class
from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate

REPO = Path(__file__).resolve().parents[4]


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(json.loads(message.data))


@pytest.fixture
def node_factory(monkeypatch):
    config = yaml.safe_load((REPO / "Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/config/default.yaml").read_text())
    override = yaml.safe_load((REPO / "scripts/InteractiveNav/configs/semantic_decision/object_goal_v3_full_mllm.yaml").read_text())
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    monkeypatch.setattr(decision.rospy, "init_node", lambda *_a, **_k: None)
    monkeypatch.setattr(decision.rospy, "get_param", lambda key, default=None: deepcopy(config.get(key.lstrip("~"), default)))
    monkeypatch.setattr(decision.rospy, "Publisher", lambda *_a, **_k: Publisher())
    monkeypatch.setattr(decision.rospy, "Subscriber", lambda *_a, **_k: None)
    monkeypatch.setattr(decision.rospy, "Timer", lambda *_a, **_k: None)
    monkeypatch.setattr(decision, "load_env_file", lambda *_a, **_k: None)
    monkeypatch.setattr(decision, "apply_model_env_overrides", lambda config: config)
    return lambda variant: decision_node_class(decision.SemanticRuleDecisionNode, variant)()


def candidate(name, distance):
    return BehaviorCandidate(
        candidate_id=name, behavior_type="INTERACT", source="interaction_graph",
        target_id=name, target_name="door", goal_xyyaw=[distance, 0., 0.],
        interaction_command={"node_id": name, "action": "open"},
        features={"distance_m": distance, "confidence": .9},
        metadata={"node_type": "portal", "state": "closed", "semantic_name": "door"},
    ).to_dict()


def snapshot(candidates):
    return {"episode_id": "ablation-test", "sequence": 1, "graph_revision": 1,
            "candidates": deepcopy(candidates), "candidate_count": len(candidates),
            "robot_xy": [-1., 0.], "graph_context": {"nodes": [], "edges": []},
            "exploration_context": {"observation_step": 100, "initial_scan_complete": True,
                                    "interaction_frontier_count": 2, "frontier_exhausted": False,
                                    "unresolved_interaction_target_count": 2,
                                    "connected_unknown_area_present": True}}


@pytest.mark.parametrize("variant", ["full", "no_interaction_graph", "no_task_decision", "no_outcome_update"])
def test_actual_node_selects_and_publishes_with_current_config(node_factory, monkeypatch, variant):
    node = node_factory(variant)
    calls = []

    def request(payload, **kwargs):
        calls.append(payload)
        return {"ranked_ids": ["far"], "reason": "INFORMATION_GAIN", "confidence": "high"}

    monkeypatch.setattr(node.model_policy, "_request", request)
    data = snapshot([candidate("near", 1.), candidate("far", 4.)])
    node.latest_candidates_payload = deepcopy(data)
    node._decide_from_snapshot(data)
    assert node.active_candidate_id == ("near" if variant == "no_task_decision" else "far")
    assert bool(calls) == (variant != "no_task_decision")
    if variant == "no_interaction_graph":
        assert "object_memory" in calls[0] and "graph" not in calls[0]
    elif calls:
        assert "graph" in calls[0]
    assert node.trace_pub.messages


def test_no_outcome_update_keeps_success_lifecycle_but_removes_continuation(node_factory):
    pending = {}
    for variant in ("full", "no_outcome_update"):
        node = node_factory(variant)
        portal = candidate("door", 1.)
        portal["portal_center_xy"] = [0., 0.]
        portal["goal_xyyaw"] = [-1., 0., 0.]
        node.latest_candidates_payload = snapshot([portal])
        node.active_candidate_id = "door"
        node.active_decision_id = "decision_1"
        node.active_behavior_type = "INTERACT"
        node.active_interaction_candidate = portal
        node._feedback_callback(String(data=json.dumps({
            "candidate_id": "door", "decision_id": "decision_1", "status": "SUCCEEDED",
            "success": True, "detail": {"event_id": "open_1", "node_id": "door", "action": "open"},
        })))
        pending[variant] = node.pending_post_interaction_traversal
        assert not node.active_candidate_id
        if variant == "no_outcome_update":
            assert not node.post_interaction_refresh_gate.active
    assert pending["full"]
    assert pending["no_outcome_update"] == {}


def test_no_outcome_update_filters_latest_snapshot_revalidation(node_factory):
    node = node_factory("no_outcome_update")
    continuation = candidate("event", 1.)
    continuation["metadata"]["post_interaction_traversal"] = True
    data = snapshot([continuation, candidate("ordinary", 2.)])
    eligible, rejected = node._eligible_candidates_from_snapshot(data, now=0., region_history={})
    assert [c.candidate_id for c in eligible] == ["ordinary"]
    assert len(data["candidates"]) == 2


def test_actual_mapping_preserves_sensor_and_occupancy_callbacks():
    from semantic_mapping_node import SemanticMappingNode
    adapted = mapping_node_class(SemanticMappingNode, "no_outcome_update")
    for name in ("gt_observation_callback", "object_callback", "occupancy_callback", "attribute_updates_callback"):
        assert getattr(adapted, name) is getattr(SemanticMappingNode, name)
    # Suppressed callbacks must not access graph_store or optimistic overlay state.
    node = adapted.__new__(adapted)
    node.interaction_command_callback(String(data='{"node_id":"door","action":"open"}'))
    node.interaction_result_callback(String(data='{"node_id":"door","success":true}'))
