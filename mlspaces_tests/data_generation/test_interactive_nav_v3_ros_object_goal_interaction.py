"""Unit coverage for evaluator-owned opaque ROS object interactions.

This deliberately uses only fakes: the public method command contains an
opaque instance ID, while raw simulator names/joints remain available only in
the evaluator-private score record.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.InteractiveNav.evaluation import benchmark_metrics, benchmark_runner
from scripts.InteractiveNav.evaluation.ros_object_goal_adapter import EvaluatorInteractionRequest
from scripts.InteractiveNav.evaluation.trusted_interaction_skill import (
    JointOpenResult,
    OpaqueObjectRegistry,
    OpenPostconditionSpec,
    TrustedInteractionSkill,
)


class _FakeAdapter:
    """Minimal evaluator-side adapter fake; it never imports or starts ROS."""

    def __init__(self, request: EvaluatorInteractionRequest) -> None:
        self._request: EvaluatorInteractionRequest | None = request
        self.completions: list[tuple[str, bool]] = []

    def pop_next_interaction_request(self) -> EvaluatorInteractionRequest | None:
        request, self._request = self._request, None
        return request

    def complete_interaction(self, command_id: str, *, success: bool) -> dict[str, str]:
        self.completions.append((command_id, success))
        return {"status": "COMPLETED" if success else "FAILED"}


def _runtime_joint(*, object_name: str, joint_name: str, joint_index: int) -> benchmark_runner.RuntimeJoint:
    return benchmark_runner.RuntimeJoint(
        object_name=object_name,
        object_category="Fridge",
        domain="container",
        joint_name=joint_name,
        joint_index=joint_index,
        body_id=1,
        aabb_center=np.asarray([0.0, 0.0, 0.0]),
        aabb_size=np.asarray([1.0, 1.0, 1.0]),
    )


def test_opaque_ros_object_command_keeps_private_resolution_out_of_public_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sealed multi-joint skill credits V3 rows without leaking their IDs."""

    opaque_id = "obj_000017"
    command_id = "request_opaque_001"
    raw_object_name = "private_fridge_body_42"
    raw_joint_names = ("private_fridge_hinge_left", "private_fridge_hinge_right")
    scoring_ids = ("v3_internal_open_left", "v3_internal_open_right")
    joints = (
        _runtime_joint(object_name=raw_object_name, joint_name=raw_joint_names[0], joint_index=11),
        _runtime_joint(object_name=raw_object_name, joint_name=raw_joint_names[1], joint_index=12),
    )
    registry = OpaqueObjectRegistry()
    registry.register(
        opaque_id,
        joints=joints,
        object_ref=raw_object_name,
        open_postcondition=OpenPostconditionSpec(success_fraction=0.8, minimum_open_joints=1),
    )
    executed_joint_names: list[str] = []

    def execute_open_joint(joint: benchmark_runner.RuntimeJoint) -> JointOpenResult:
        # These names are deliberately private to the trusted executor.
        executed_joint_names.append(joint.joint_name)
        result_by_index = {
            11: JointOpenResult(True, 0.0, 0.90, simulated_seconds=0.35),
            12: JointOpenResult(True, 0.0, 0.85, simulated_seconds=0.65),
        }
        return result_by_index[joint.joint_index]

    adapter = _FakeAdapter(
        EvaluatorInteractionRequest(
            command_id=command_id,
            episode_id="episode_public_1",
            instance_id=opaque_id,
            action="open",
            private_handle=object(),
        )
    )
    runtime = SimpleNamespace(
        adapter=adapter,
        skill=TrustedInteractionSkill(registry, execute_open_joint),
        opaque_to_source_name={opaque_id: raw_object_name},
        opaque_to_joints={opaque_id: joints},
    )
    task = SimpleNamespace(env=object(), get_observations=lambda: {"camera": "public-observation"})
    config = SimpleNamespace(
        interaction_max_distance_m=1.75,
        require_interaction_visible=True,
        record_video=False,
    )
    episode = {
        "interactive_nav": {
            "interaction_requirement": "required",
            "interactions": [
                {
                    "interaction_id": scoring_ids[0],
                    "object_name": raw_object_name,
                    "joint_index": 11,
                    "prerequisites": [],
                },
                {
                    "interaction_id": scoring_ids[1],
                    "object_name": raw_object_name,
                    "joint_index": 12,
                    "prerequisites": [],
                },
            ],
            "oracle_plans": [
                {
                    "plan_id": "private_full_open_plan",
                    "required_interaction_ids": list(scoring_ids),
                }
            ],
        }
    }
    published_steps: list[int] = []
    monkeypatch.setattr(
        benchmark_runner,
        "_check_interaction_access",
        lambda *_args, **_kwargs: (True, {"distance_m": 0.5, "visibility": 1.0}),
    )
    monkeypatch.setattr(benchmark_runner, "_capture_head_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        benchmark_runner,
        "_publish_restricted_ros_frame",
        lambda _runtime, _task, *, decision_index: published_steps.append(decision_index) or True,
    )
    monkeypatch.setattr(benchmark_runner, "_discard_task_rollout_cache", lambda _task: None)

    consumed = benchmark_runner._consume_pending_ros_object_goal_interaction(
        task=task,
        runtime=runtime,
        episode=episode,
        private_attempts=[],
        config=config,
        decision_index=8,
        frames=[],
    )

    assert consumed is not None
    private_attempt = consumed["private_attempt"]
    public_attempt = consumed["public_attempt"]
    assert executed_joint_names == list(raw_joint_names)
    assert adapter.completions == [(command_id, True)]
    assert private_attempt["resolved_object_name"] == raw_object_name
    assert private_attempt["resolved_interaction_ids"] == list(scoring_ids)
    assert private_attempt["resolved_interaction_id"] == scoring_ids[0]
    assert private_attempt["simulated_seconds"] == pytest.approx(1.0)
    assert consumed["simulated_seconds"] == pytest.approx(1.0)
    assert published_steps == [8]

    # This is the policy-visible projection.  It carries only opaque routing,
    # high-level outcome and elapsed skill time; raw object/joint/V3 IDs remain
    # exclusively in ``private_attempt`` above.
    assert public_attempt == {
        "request_id": command_id,
        "instance_id": opaque_id,
        "operation": "open",
        "status": "completed",
        "decision_step": 8,
        "simulated_seconds": pytest.approx(1.0),
        "result_status": "COMPLETED",
    }
    public_json = json.dumps(public_attempt, sort_keys=True)
    for private_value in (raw_object_name, *raw_joint_names, *scoring_ids):
        assert private_value not in public_json
    assert {"resolved_object_name", "resolved_joint_name", "resolved_interaction_id"}.isdisjoint(public_attempt)

    # The plural private IDs are what allow one sealed object action to satisfy
    # its V3 multi-joint plan.  Patch final joint fractions only; no simulator is
    # created for this regression test.
    monkeypatch.setattr(benchmark_metrics, "joint_open_fraction", lambda _env, _row: 1.0)
    terminal = benchmark_metrics.score_interactions(object(), episode, [private_attempt])
    assert terminal.valid_plan_id == "private_full_open_plan"
    assert terminal.required_interaction_success is True
    assert terminal.correct_action_count == 2
