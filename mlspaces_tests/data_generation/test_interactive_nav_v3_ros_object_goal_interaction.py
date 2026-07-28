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
    ObjectInteractionResult,
    OpaqueObjectRegistry,
    OpenPostcondition,
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


def test_drawer_scan_gate_and_scoring_ids_use_private_drawer_type_and_physical_order() -> None:
    source_name = "private_dresser_body"
    top = _runtime_joint(object_name=source_name, joint_name="top_slide", joint_index=9)
    bottom = _runtime_joint(object_name=source_name, joint_name="bottom_slide", joint_index=4)
    # Deliberately store the V3 rows in the opposite order.  The macro's
    # top-to-bottom physical scan order must be what prerequisite scoring sees.
    episode = {
        "interactive_nav": {
            "interactions": [
                {
                    "interaction_id": "drawer_bottom",
                    "object_name": source_name,
                    "joint_index": 4,
                    "type": "container_sliding_drawer",
                },
                {
                    "interaction_id": "drawer_top",
                    "object_name": source_name,
                    "joint_index": 9,
                    "type": "container_sliding_drawer",
                },
            ]
        }
    }

    assert benchmark_runner._is_trusted_drawer_scan_target(episode, source_name, (top, bottom))
    assert benchmark_runner._trusted_drawer_scan_joint_indices(episode, source_name) == {4, 9}
    assert benchmark_runner._successful_drawer_scan_interaction_ids(
        episode=episode,
        source_name=source_name,
        opened_joints=(top, bottom),
    ) == ["drawer_top", "drawer_bottom"]

    episode["interactive_nav"]["interactions"][0]["type"] = "container_hinged_door"
    episode["interactive_nav"]["interactions"][1]["type"] = "container_hinged_door"
    assert not benchmark_runner._is_trusted_drawer_scan_target(episode, source_name, (top, bottom))


def test_object_skill_completion_requires_a_complete_object_local_oracle_plan() -> None:
    source_name = "private_fridge_body"
    episode = {
        "interactive_nav": {
            "interactions": [
                {
                    "interaction_id": "outer",
                    "object_name": source_name,
                    "joint_index": 3,
                },
                {
                    "interaction_id": "inner",
                    "object_name": source_name,
                    "joint_index": 1,
                    "prerequisites": [{"interaction_id": "outer"}],
                },
                {
                    "interaction_id": "alternative",
                    "object_name": source_name,
                    "joint_index": 0,
                },
            ],
            "oracle_plans": [
                {"plan_id": "two_joint", "required_interaction_ids": ["outer", "inner"]},
                {"plan_id": "alternative", "required_interaction_ids": ["alternative"]},
            ],
        }
    }

    assert not benchmark_runner._object_skill_satisfies_an_oracle_plan(
        episode=episode,
        source_name=source_name,
        successful_ids=["outer"],
    )
    assert benchmark_runner._object_skill_satisfies_an_oracle_plan(
        episode=episode,
        source_name=source_name,
        successful_ids=["inner", "outer"],
    )
    assert benchmark_runner._object_skill_satisfies_an_oracle_plan(
        episode=episode,
        source_name=source_name,
        successful_ids=["alternative"],
    )
    assert benchmark_runner._object_skill_satisfies_an_oracle_plan(
        episode={"interactive_nav": {"interactions": [], "oracle_plans": []}},
        source_name=source_name,
        successful_ids=[],
    )
    assert benchmark_runner._order_interaction_ids_by_oracle_plan(
        episode,
        ["inner", "outer"],
    ) == ["outer", "inner"]
    inner_joint = _runtime_joint(
        object_name=source_name,
        joint_name="inner",
        joint_index=1,
    )
    outer_joint = _runtime_joint(
        object_name=source_name,
        joint_name="outer",
        joint_index=3,
    )
    ordered_joints = benchmark_runner._ordered_object_skill_joints(
        source_name=source_name,
        all_joints=(inner_joint, outer_joint),
        interactions=episode["interactive_nav"]["interactions"],
        plans=episode["interactive_nav"]["oracle_plans"],
    )
    assert [joint.joint_index for joint in ordered_joints] == [3, 1]


def test_partial_object_skill_result_is_reported_failed_to_ros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_name = "private_fridge_body"
    opaque_id = "obj_000008"
    command_id = "partial_open"
    joints = (
        _runtime_joint(object_name=source_name, joint_name="outer", joint_index=3),
        _runtime_joint(object_name=source_name, joint_name="inner", joint_index=1),
    )
    adapter = _FakeAdapter(
        EvaluatorInteractionRequest(
            command_id=command_id,
            episode_id="episode_000008",
            instance_id=opaque_id,
            action="open",
            private_handle=object(),
        )
    )
    public_result = ObjectInteractionResult(
        request_id=command_id,
        instance_id=opaque_id,
        operation="open",
        status="completed",
    )
    runtime = SimpleNamespace(
        adapter=adapter,
        skill=SimpleNamespace(
            execute_private=lambda _request: SimpleNamespace(
                joint_results=(
                    JointOpenResult(True, 0.0, 0.9),
                    JointOpenResult(False, 0.0, 0.2),
                ),
                public_result=public_result,
                postcondition=OpenPostcondition.SATISFIED,
            )
        ),
        opaque_to_source_name={opaque_id: source_name},
        opaque_to_joints={opaque_id: joints},
    )
    episode = {
        "interactive_nav": {
            "interaction_requirement": "required",
            "interactions": [
                {
                    "interaction_id": "outer",
                    "object_name": source_name,
                    "joint_index": 3,
                    "prerequisites": [],
                },
                {
                    "interaction_id": "inner",
                    "object_name": source_name,
                    "joint_index": 1,
                    "prerequisites": [{"interaction_id": "outer"}],
                },
            ],
            "oracle_plans": [
                {"plan_id": "both", "required_interaction_ids": ["outer", "inner"]}
            ],
        }
    }
    task = SimpleNamespace(env=object(), get_observations=lambda: {})
    config = SimpleNamespace(
        interaction_max_distance_m=1.75,
        require_interaction_visible=True,
        record_video=False,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_check_interaction_access",
        lambda *_args, **_kwargs: (True, {}),
    )
    monkeypatch.setattr(benchmark_runner, "_capture_head_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        benchmark_runner,
        "_publish_restricted_ros_frame",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(benchmark_runner, "_discard_task_rollout_cache", lambda _task: None)

    consumed = benchmark_runner._consume_pending_ros_object_goal_interaction(
        task=task,
        runtime=runtime,
        episode=episode,
        private_attempts=[],
        config=config,
        decision_index=1,
        frames=[],
    )

    assert consumed is not None
    assert adapter.completions == [(command_id, False)]
    assert consumed["public_attempt"]["status"] == "failed"
    assert consumed["private_attempt"]["success"] is False
    assert consumed["private_attempt"]["resolved_interaction_ids"] == ["outer"]


def test_failed_drawer_scan_cannot_return_transient_target_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opaque_id = "obj_000071"
    source_name = "private_dresser_body"
    joint = _runtime_joint(object_name=source_name, joint_name="private_slide", joint_index=4)
    command_id = "drawer_scan_failure"
    adapter = _FakeAdapter(
        EvaluatorInteractionRequest(
            command_id=command_id,
            episode_id="episode_public_2",
            instance_id=opaque_id,
            action="open",
            private_handle=object(),
            sequence_type="drawer_scan",
            open_regions=((0.5, 0.2),),
        )
    )
    runtime = SimpleNamespace(
        adapter=adapter,
        skill=object(),
        opaque_to_source_name={opaque_id: source_name},
        opaque_to_joints={opaque_id: (joint,)},
    )
    episode = {
        "interactive_nav": {
            "interaction_requirement": "required",
            "interactions": [
                {
                    "interaction_id": "drawer_target",
                    "object_name": source_name,
                    "joint_index": 4,
                    "type": "container_sliding_drawer",
                    "prerequisites": [],
                }
            ],
            "oracle_plans": [],
        }
    }
    task = SimpleNamespace(env=object(), get_observations=lambda: {"camera": "public-observation"})
    config = SimpleNamespace(
        interaction_max_distance_m=1.75,
        require_interaction_visible=True,
        record_video=False,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_check_interaction_access",
        lambda *_args, **_kwargs: (True, {"distance_m": 0.5, "visibility": 1.0}),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_execute_private_drawer_scan",
        lambda **_kwargs: {
            "success": False,
            "joint_results": (),
            "opened_joints": (),
            "simulated_seconds": 1.0,
            # Simulate a buggy future executor.  The V3 boundary must still
            # reject its transient evidence when the macro has failed.
            "target_discovery": {
                "distance_m": 0.4,
                "visibility_fraction": 0.1,
                "visible_pixels": 32,
                "group_index": 0,
            },
            "metadata": {},
            "observation": {"camera": "public-observation"},
        },
    )
    monkeypatch.setattr(benchmark_runner, "_capture_head_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark_runner, "_publish_restricted_ros_frame", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(benchmark_runner, "_discard_task_rollout_cache", lambda _task: None)

    consumed = benchmark_runner._consume_pending_ros_object_goal_interaction(
        task=task,
        runtime=runtime,
        episode=episode,
        private_attempts=[],
        config=config,
        decision_index=4,
        frames=[],
    )

    assert consumed is not None
    assert adapter.completions == [(command_id, False)]
    assert consumed["private_attempt"]["success"] is False
    assert consumed["target_discovery"] is None


@pytest.mark.parametrize(
    ("direct_bbox_drawer_scan", "expected_fallback"),
    [(False, False), (True, True)],
)
def test_empty_drawer_regions_fall_back_only_for_direct_public_bbox_scan(
    monkeypatch: pytest.MonkeyPatch,
    direct_bbox_drawer_scan: bool,
    expected_fallback: bool,
) -> None:
    opaque_id = "obj_000071"
    source_name = "private_dresser_body"
    joint = _runtime_joint(object_name=source_name, joint_name="private_slide", joint_index=4)
    command_id = f"drawer_scan_empty_{int(direct_bbox_drawer_scan)}"
    adapter = _FakeAdapter(
        EvaluatorInteractionRequest(
            command_id=command_id,
            episode_id="episode_public_3",
            instance_id=opaque_id,
            action="open",
            private_handle=object(),
            sequence_type="drawer_scan",
            open_regions=(),
            direct_bbox_drawer_scan=direct_bbox_drawer_scan,
        )
    )
    runtime = SimpleNamespace(
        adapter=adapter,
        skill=object(),
        opaque_to_source_name={opaque_id: source_name},
        opaque_to_joints={opaque_id: (joint,)},
    )
    episode = {
        "interactive_nav": {
            "interaction_requirement": "required",
            "interactions": [
                {
                    "interaction_id": "drawer_target",
                    "object_name": source_name,
                    "joint_index": 4,
                    "type": "container_sliding_drawer",
                    "prerequisites": [],
                }
            ],
            "oracle_plans": [],
        }
    }
    task = SimpleNamespace(env=object(), get_observations=lambda: {"camera": "public-observation"})
    config = SimpleNamespace(
        interaction_max_distance_m=1.75,
        require_interaction_visible=True,
        record_video=False,
    )
    scan_kwargs: dict = {}
    monkeypatch.setattr(
        benchmark_runner,
        "_check_interaction_access",
        lambda *_args, **_kwargs: (True, {"distance_m": 0.5, "visibility": 1.0}),
    )

    def execute_scan(**kwargs):
        scan_kwargs.update(kwargs)
        success = bool(kwargs["fallback_to_all"])
        return {
            "success": success,
            "joint_results": (),
            "opened_joints": (joint,) if success else (),
            "simulated_seconds": 1.0 if success else 0.0,
            "target_discovery": None,
            "metadata": {"execution_mode": "drawer_scan"},
            "observation": {"camera": "public-observation"},
        }

    monkeypatch.setattr(benchmark_runner, "_execute_private_drawer_scan", execute_scan)
    monkeypatch.setattr(benchmark_runner, "_capture_head_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark_runner, "_publish_restricted_ros_frame", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(benchmark_runner, "_discard_task_rollout_cache", lambda _task: None)

    consumed = benchmark_runner._consume_pending_ros_object_goal_interaction(
        task=task,
        runtime=runtime,
        episode=episode,
        private_attempts=[],
        config=config,
        decision_index=5,
        frames=[],
    )

    assert consumed is not None
    assert scan_kwargs["open_regions"] == ()
    assert scan_kwargs["fallback_to_all"] is expected_fallback
    assert consumed["private_attempt"]["success"] is expected_fallback
    assert adapter.completions == [(command_id, expected_fallback)]
