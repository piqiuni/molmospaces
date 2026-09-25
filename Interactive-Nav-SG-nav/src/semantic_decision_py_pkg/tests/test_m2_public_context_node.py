from collections import deque
from pathlib import Path
import sys
from threading import RLock
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for path in (SCRIPTS, SCRIPTS.parents[1] / "semantic_mllm_py_pkg" / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

pytest.importorskip("rospy")
from semantic_rule_decision_node import SemanticRuleDecisionNode
from semantic_candidate_node import SemanticCandidateNode
import semantic_candidate_node as candidate_module


def test_online_robot_trajectory_preserves_revisits_without_selected_goal_inference():
    node = SemanticRuleDecisionNode.__new__(SemanticRuleDecisionNode)
    node.state_lock = RLock()
    node.entered_room_ids = set()
    node.initial_robot_xy = None
    node.initial_position_source = {}
    node.room_visit_history = []
    graph = {"nodes": [
        {"id": "room_1", "type": "room", "room_id": 1, "aabb_center": [0, 0, 0], "aabb_size": [2, 2, 1]},
        {"id": "room_2", "type": "room", "room_id": 2, "aabb_center": [4, 0, 0], "aabb_size": [2, 2, 1]},
    ]}
    for step, xy in enumerate(([0, 0], [0.2, 0], [4, 0], [0, 0])):
        snapshot = {
            "sequence": step + 1, "robot_xy": xy, "graph_context": graph,
            "exploration_context": {"observation_step": step},
            "candidates": [{"metadata": {"target_room_id": 99}}],
        }
        node._update_entered_rooms(snapshot)
    context = node._trajectory_context(snapshot)
    assert context["initial_xy"] == [0, 0]
    assert context["initial_position_source"] == {
        "kind": "first_candidate_observation", "observation_step": 0, "candidate_sequence": 1,
    }
    assert [v["room_id"] for v in context["room_visit_history"]] == ["room_1", "room_2", "room_1"]
    assert [v["entry_step"] for v in context["room_visit_history"]] == [0, 2, 3]
    snapshot["exploration_context"]["observation_step"] = 2
    assert len(node._trajectory_context(snapshot)["room_visit_history"]) == 2


def test_online_history_exports_thirty_decisions_with_outcomes_and_goals():
    node = SemanticRuleDecisionNode.__new__(SemanticRuleDecisionNode)
    node.state_lock = RLock()
    node._refresh_history_metrics = lambda _snapshot: None
    node.decision_history = deque([
        {
            "decision_id": f"d{step}", "observation_step": step,
            "candidate_id": f"frontier:{step}", "target_id": "target",
            "target_room_id": 2, "goal_xy": [step, 0],
            "result": "FAILED", "failure_reason": "navigation_unreachable",
        }
        for step in range(32)
    ], maxlen=32)
    node.group_history = {}
    history, _ = node._history_context({"exploration_context": {"observation_step": 32}})
    assert len(history) == 30
    assert history[0]["decision_id"] == "d2"
    assert history[-1]["step"] == 31
    assert history[-1]["goal_xy"] == [31, 0]
    assert history[-1]["target_room_id"] == 2
    assert history[-1]["failure_reason"] == "navigation_unreachable"


def test_candidate_producer_keeps_observed_status_even_with_proposal_stream():
    node = SemanticCandidateNode.__new__(SemanticCandidateNode)
    node.has_proposal_stream = True
    node.explorer_status = {}
    node._explorer_callback(SimpleNamespace(data='{"ready":true,"frontier_count":0,"frontier_clusters":[]}'))
    assert node.explorer_status["frontier_clusters"] == []
    assert node.explorer_status_received_ts > 0


@pytest.mark.parametrize("limit", [0, 8, 30])
def test_online_history_window_uses_experiment_config_without_erasing_memory(limit):
    node = SemanticRuleDecisionNode.__new__(SemanticRuleDecisionNode)
    node.state_lock = RLock()
    node.model_policy = SimpleNamespace(config=SimpleNamespace(recent_decision_limit=limit))
    node._refresh_history_metrics = lambda _snapshot: None
    node.decision_history = deque([
        {"decision_id": f"d{step}", "observation_step": step} for step in range(32)
    ], maxlen=32)
    node.group_history = {}
    history, _ = node._history_context({"exploration_context": {"observation_step": 32}})
    assert len(history) == limit
    assert len(node.decision_history) == 32
    if limit:
        assert history[0]["decision_id"] == f"d{32-limit}"


def test_candidate_position_frame_is_raw_odometry_frame_without_coordinate_change():
    node = SemanticCandidateNode.__new__(SemanticCandidateNode)
    node.map_frame = "map"
    message = SimpleNamespace(header=SimpleNamespace(frame_id="odom"), pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=3, y=4))))
    node._odom_callback(message)
    assert node.robot_position_frame_id == "odom"
    assert node.robot_xy == (3, 4)
    message.header.frame_id = ""
    node._odom_callback(message)
    assert node.robot_position_frame_id == ""
    assert node.robot_frame == "map"  # Existing execution fallback is unchanged.


def test_online_model_pose_tf_does_not_modify_execution_pose():
    node = SemanticCandidateNode.__new__(SemanticCandidateNode)
    node.robot_xy = (1, 2)
    node.robot_z = 0
    node.robot_position_frame_id = "odom"
    node.robot_pose_stamp = SimpleNamespace(to_sec=lambda: 12)
    calls = []
    def lookup(target, source, stamp):
        calls.append((target, source, stamp))
        return ([10, -3, 0], [0, 0, 0, 1])
    node.tf_listener = SimpleNamespace(lookupTransform=lookup)
    result = node._model_robot_pose_context({"frame_id": "map"})
    assert result["robot_graph_xy"] == [11, -1]
    assert node.robot_xy == (1, 2)
    assert calls[0] == ("map", "odom", node.robot_pose_stamp)


def test_online_missing_tf_keeps_graph_pose_unknown():
    node = SemanticCandidateNode.__new__(SemanticCandidateNode)
    node.robot_xy = (1, 2)
    node.robot_position_frame_id = "odom"
    def unavailable(*_args):
        raise candidate_module.tf.LookupException("not yet available")
    node.tf_listener = SimpleNamespace(lookupTransform=unavailable)
    result = node._model_robot_pose_context({"frame_id": "map"})
    assert result["robot_graph_xy"] is None
    assert result["robot_graph_pose_source"]["reason"] == "transform_unavailable:LookupException"
    assert node.robot_xy == (1, 2)


def test_online_initial_and_room_history_wait_for_valid_graph_pose():
    node = SemanticRuleDecisionNode.__new__(SemanticRuleDecisionNode)
    node.state_lock = RLock()
    node.entered_room_ids = set()
    node.initial_robot_xy = None
    node.initial_position_source = {}
    node.room_visit_history = []
    snapshot = {
        "robot_xy": [0, 0], "robot_position_frame_id": "odom", "robot_graph_xy": None,
        "robot_graph_frame_id": "map", "graph_context": {"frame_id": "map", "nodes": [
            {"id": "room_2", "type": "room", "room_id": 2, "aabb_center": [10, 0, 0], "aabb_size": [2, 2, 1]},
        ]}, "exploration_context": {"observation_step": 1},
    }
    node._update_entered_rooms(snapshot)
    assert node.initial_robot_xy is None
    assert node.room_visit_history == []
    snapshot["robot_graph_xy"] = [10, 0]
    node._update_entered_rooms(snapshot)
    assert node.initial_robot_xy == [10, 0]
    assert node.initial_position_source["frame_id"] == "map"
    assert node.room_visit_history[0]["room_id"] == "room_2"


def test_pending_history_does_not_claim_zero_frontier_change():
    node = SemanticRuleDecisionNode.__new__(SemanticRuleDecisionNode)
    node.state_lock = RLock()
    node._refresh_history_metrics = lambda _snapshot: None
    node.group_history = {}
    node.decision_history = deque([
        {"decision_id": "pending", "result": "PENDING", "frontier_length_delta_m": 0, "frontier_metrics_evaluated": False},
        {"decision_id": "evaluated", "result": "SUCCEEDED", "frontier_length_delta_m": 0, "frontier_shrink_m": 0, "frontier_metrics_evaluated": True},
    ])
    history, _ = node._history_context({})
    assert "frontier_length_delta_m" not in history[0]
    assert history[1]["frontier_length_delta_m"] == 0
