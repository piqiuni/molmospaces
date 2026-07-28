from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

from std_msgs.msg import String


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from semantic_decision_py_pkg.mission_completion import TargetMissionTracker
from semantic_rule_decision_node import SemanticRuleDecisionNode


class FakePublisher:
    def __init__(self) -> None:
        self.payloads = []

    def publish(self, message: String) -> None:
        self.payloads.append(json.loads(message.data))


def target_candidate(candidate_id: str, distance_m: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "behavior_type": "NAVIGATE",
        "features": {"distance_m": distance_m},
        "metadata": {
            "target_goal": True,
            "target_reliably_observed": True,
            "target_visible_now": True,
        },
    }


def make_node(cooldown_until: dict[str, float]) -> SemanticRuleDecisionNode:
    node = object.__new__(SemanticRuleDecisionNode)
    node.state_lock = threading.RLock()
    node.latest_candidates_payload = {
        "episode_id": "episode_1",
        "target_context": {"enabled": True},
    }
    node.target_context = {"enabled": True}
    node.episode_active = True
    node.lifecycle_generation = 0
    node.target_mission = TargetMissionTracker()
    node.cooldown_until = dict(cooldown_until)
    node.active_candidate_id = "frontier:active"
    node.active_decision_id = "decision_000001"
    node.active_behavior_type = "EXPLORE"
    node.active_interaction_candidate = {}
    node.active_target_goal = False
    node.preempt_requested_for_decision_id = ""
    node.priority_target_candidate_id = ""
    node.preempt_pub = FakePublisher()
    node._publish_goal_status = lambda *_args, **_kwargs: None
    return node


def send_candidates(
    node: SemanticRuleDecisionNode,
    candidates: list[dict],
    target_context: dict | None = None,
) -> None:
    node._candidate_callback(
        String(
            data=json.dumps(
                {
                    "episode_id": "episode_1",
                    "sequence": 2,
                    "target_context": target_context or {"enabled": True},
                    "candidates": candidates,
                }
            )
        )
    )


def test_cooled_target_does_not_preempt_active_explore() -> None:
    candidate_id = "target:television"
    node = make_node({candidate_id: time.monotonic() + 100.0})

    send_candidates(node, [target_candidate(candidate_id, 1.0)])

    assert node.preempt_pub.payloads == []
    assert node.priority_target_candidate_id == ""


def test_target_preempts_after_cooldown_expires() -> None:
    candidate_id = "target:television"
    node = make_node({candidate_id: time.monotonic() - 1.0})

    send_candidates(node, [target_candidate(candidate_id, 1.0)])

    assert len(node.preempt_pub.payloads) == 1
    assert node.preempt_pub.payloads[0]["replacement_candidate_id"] == candidate_id


def test_available_second_target_preempts_when_best_raw_target_is_cooled() -> None:
    cooled_id = "target:near"
    available_id = "target:far"
    node = make_node({cooled_id: time.monotonic() + 100.0})

    send_candidates(
        node,
        [
            target_candidate(cooled_id, 0.5),
            target_candidate(available_id, 1.5),
        ],
    )

    assert len(node.preempt_pub.payloads) == 1
    assert node.preempt_pub.payloads[0]["replacement_candidate_id"] == available_id


def test_only_one_preempt_is_published_for_an_active_decision() -> None:
    candidate_id = "target:television"
    node = make_node({})

    send_candidates(node, [target_candidate(candidate_id, 1.0)])
    send_candidates(node, [target_candidate(candidate_id, 1.0)])

    assert len(node.preempt_pub.payloads) == 1


def test_inactive_episode_clears_target_preemption_state() -> None:
    candidate_id = "target:television"
    node = make_node({})

    send_candidates(
        node,
        [target_candidate(candidate_id, 1.0)],
        target_context={
            "enabled": True,
            "episode_active": False,
            "episode_generation": 1,
        },
    )

    assert node.episode_active is False
    assert node.priority_target_candidate_id == ""
    assert node.preempt_pub.payloads == []
    assert node.active_candidate_id == ""


def test_lifecycle_snapshot_rejects_stale_generation() -> None:
    node = make_node({})
    node.lifecycle_generation = 3

    assert node._lifecycle_snapshot_is_current(
        {"target_context": {"episode_active": True, "episode_generation": 3}}
    )
    assert not node._lifecycle_snapshot_is_current(
        {"target_context": {"episode_active": True, "episode_generation": 2}}
    )
    assert not node._lifecycle_snapshot_is_current(
        {"target_context": {"episode_active": False, "episode_generation": 3}}
    )
