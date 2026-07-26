from __future__ import annotations

import numpy as np

from scripts.InteractiveNav.evaluation.metrics import summarise_results
from scripts.InteractiveNav.evaluation.policies import MolmoSpacesPolicyAdapter, ScriptedOraclePolicy
from scripts.InteractiveNav.evaluation.runner import EvaluationConfig
from scripts.InteractiveNav.evaluation.types import PolicyObservation


def test_summary_has_domain_and_requirement_groups() -> None:
    row = {
        "domains": ["channel"],
        "interaction_requirement": "required",
        "success": True,
        "nav_success": True,
        "required_interaction_success": True,
        "sequence_success": True,
        "non_interaction_success": None,
        "step_count": 3,
        "navigation_path_length_m": 1.5,
        "wrong_interaction_count": 0,
        "terminal_reason": "nav_success",
    }
    summary = summarise_results([row])
    assert summary["groups"]["overall"]["success_rate"] == 1.0
    assert summary["groups"]["domain/channel"]["episode_count"] == 1
    assert summary["groups"]["requirement/required"]["sequence_success_rate"] == 1.0


def test_oracle_policy_is_explicitly_gt_only() -> None:
    policy = ScriptedOraclePolicy()
    policy.reset({"_oracle_steps": [{"type": "open_joint", "interaction_id": "i", "object_name": "obj", "joint_index": 2}]})
    action = policy.act(PolicyObservation(None, "find obj", 0, 0.0, None))
    assert policy.uses_oracle_gt is True
    assert action.kind == "interact"
    assert action.metadata["oracle_interaction_id"] == "i"


def test_oracle_navigation_waits_for_live_pose_before_advancing() -> None:
    policy = ScriptedOraclePolicy()
    policy.reset(
        {
            "_oracle_steps": [
                {
                    "type": "navigate",
                    "goal_point": [1.0, 2.0, 0.0],
                    "goal_yaw": 0.5,
                    "position_tolerance_m": 0.25,
                    "yaw_tolerance_rad": 0.2,
                },
                {"type": "open_joint", "interaction_id": "i", "object_name": "obj", "joint_index": 2},
            ]
        }
    )
    observation = PolicyObservation(None, "find obj", 0, 0.0, None)
    navigate = policy.act(observation)
    assert navigate.kind == "base"
    policy.notify_action_result(navigate, base_pose=np.array([0.0, 0.0, 0.0]))
    assert policy.act(observation).kind == "base"
    policy.notify_action_result(navigate, base_pose=np.array([1.1, 2.1, 0.55]))
    assert policy.act(observation).kind == "interact"


def test_ros_bridge_rejects_multiprocessing() -> None:
    config = EvaluationConfig(benchmark="benchmark.json", output_dir="out", policy="ros_bridge", workers=2)
    try:
        config.validate()
    except ValueError as exc:
        assert "workers 1" in str(exc)
    else:
        raise AssertionError("expected ros_bridge worker validation failure")


def test_wrapped_policy_interaction_request_uses_observed_object_not_oracle_id() -> None:
    class ExternalPolicy:
        def reset(self) -> None:
            return None

        def get_action(self, observation):
            del observation
            return {
                "kind": "interact",
                "object_name": "visible_fridge",
                "joint_index": 1,
                "operation": "open",
            }

    action = MolmoSpacesPolicyAdapter(ExternalPolicy()).act(
        PolicyObservation(None, "open the fridge", 0, 0.0, None)
    )
    assert action.kind == "interact"
    assert action.object_name == "visible_fridge"
    assert action.joint_index == 1
    assert "oracle_interaction_id" not in action.metadata
