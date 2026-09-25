"""Regression for erased visual votes and repeated actions on stale M1 state."""

from copy import deepcopy
import json
import threading
from types import SimpleNamespace

from ablations.perception import PostActionVisualGuard, inference_node_class
from semantic_mapping_py_pkg.portal_state_consensus import PortalStateConsensus
from semantic_decision_py_pkg.behavior_execution import (
    BehaviorExecutionStateMachine, STATE_WAITING_FOR_INTERACTION_OBSERVATION,
)


def inference_node():
    node = inference_node_class(type("Base", (), {}))()
    node.lock = threading.Lock()
    node.aliases = {"door_alias": "door"}
    for name in ("generations", "completed", "last_request", "pending",
                 "target_visual_history", "portal_observation_streaks"):
        setattr(node, name, {})
    node.request_queue = set()
    node.portal_state_consensus = PortalStateConsensus(cooldown_steps=60)
    return node


def result(node, **payload):
    node._interaction_result_callback(SimpleNamespace(data=json.dumps(payload)))


def test_observation_only_completion_preserves_three_distinct_visual_votes():
    node = inference_node()
    result(node, object_id="door_alias", event_id="physical1", step=10, post_state="closed")
    accepted = []
    for step, x in [(11, 0.), (12, .5), (13, 1.)]:
        vote = node.portal_state_consensus.observe(
            "door", "open", capture_step=step, observation_pose_xyyaw=[x, 0., 0.])
        accepted.append(vote["accepted"])
        result(node, object_id="door_alias", action_executed=False,
               observation_outcome="finish_without_action", post_state="closed")
    assert accepted == [False, False, True]
    assert node.generations["door"] == 1
    assert node.portal_state_consensus.cached_stable_result("door", capture_step=14)["stable_state"] == "open"


def test_duplicate_physical_completion_cannot_clear_new_votes():
    node = inference_node()
    payload = dict(object_id="door", event_id="physical1", step=10)
    result(node, **payload)
    node.portal_state_consensus.observe("door", "open", capture_step=11, observation_pose_xyyaw=[0, 0, 0])
    result(node, **payload)
    assert node.generations["door"] == 1
    assert node.portal_state_consensus.can_request("door", capture_step=12, observation_pose_xyyaw=[0, 0, 0])[1] == "duplicate_view_pose"


def candidate():
    return {"candidate_id": "interaction:door:open", "behavior_type": "INTERACT",
            "target_id": "door", "goal_xyyaw": [0., 0., 0.],
            "interaction_command": {"node_id": "door", "object_id": "door_alias", "action": "open"},
            "metadata": {"node_type": "portal", "state": "closed"}}


def test_retry_requires_post_action_accepted_visual_state_not_timer_or_result_label():
    guard = PostActionVisualGuard()
    guard.record({"object_id": "door_alias", "step": 100, "post_state": "open", "success": True})
    source = {"candidates": [candidate()]}
    for step, ready in [(90, True), (100, True), (101, False), (101, True)]:
        source["candidates"][0]["metadata"].update(perception_visual_step=step, perception_visual_ready=ready)
        before = deepcopy(source)
        updated = guard.apply(source)["candidates"][0]
        assert bool(updated["metadata"].get("observation_only_reobserve")) == (step <= 100 or not ready)
        assert source == before
    guard.record({"object_id": "door_alias", "step": 200, "action_executed": False})
    assert not guard.apply(source)["candidates"][0]["metadata"].get("observation_only_reobserve")


def test_gate_resets_between_episodes():
    guard = PostActionVisualGuard()
    guard.record({"object_id": "door", "step": 100, "episode_id": "a"})
    updated = guard.apply({"episode_id": "b", "candidates": [candidate()]})
    assert not updated["candidates"][0]["metadata"].get("observation_only_reobserve")


def test_real_execution_cannot_promote_post_action_reobserve_to_physical_open():
    guard = PostActionVisualGuard()
    guard.record({"node_id": "door", "step": 100})
    selected = guard.apply({"candidates": [candidate()]})["candidates"][0]
    machine = BehaviorExecutionStateMachine()
    machine.candidate = selected
    machine.state = STATE_WAITING_FOR_INTERACTION_OBSERVATION
    selected["metadata"].update(interaction_position_reached=True)
    assert not machine._should_fallback_to_interaction_after_m1_failure(
        {"attribute_status": "timeout"}, selected["metadata"])
    commands = machine.on_interaction_observation_result(
        {"attribute_status": "ready", "attribute_source": "mllm_attribute_inference",
         "is_currently_visible": True, "coarse_state": "closed", "observation_capture_step": 101},
        now=1., task_step_index=101)
    assert commands and all(c["kind"] != "interact" for c in commands)
    assert commands[-1]["detail"]["action_executed"] is False
