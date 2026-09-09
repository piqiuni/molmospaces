"""Unit tests for evaluator-owned opaque interaction skills.

These tests use no ROS, MuJoCo, scene, or Interactive-Nav policy code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from scripts.InteractiveNav.evaluation.trusted_interaction_skill import (
    JointOpenResult,
    ObjectInteractionRequest,
    OpaqueObjectRegistry,
    OpenPostcondition,
    OpenPostconditionSpec,
    TrustedInteractionSkill,
    classify_open_postcondition,
    redact_interaction_event,
)


@dataclass(frozen=True)
class _FakeRuntimeJoint:
    source_object_name: str
    joint_name: str
    joint_index: int


def test_object_level_open_uses_private_joints_and_redacts_all_runtime_names() -> None:
    private_joint = _FakeRuntimeJoint(
        source_object_name="private_fridge_which_must_not_escape",
        joint_name="private_hinge_joint",
        joint_index=73,
    )
    registry = OpaqueObjectRegistry()
    registry.register("obj_000017", joints=[private_joint])
    called: list[_FakeRuntimeJoint] = []

    def execute_open_joint(joint: _FakeRuntimeJoint) -> JointOpenResult:
        called.append(joint)
        return JointOpenResult(True, 0.0, 1.0, simulated_seconds=2.0)

    skill = TrustedInteractionSkill(registry, execute_open_joint)
    request = ObjectInteractionRequest(request_id="request_1", instance_id="obj_000017")
    event = skill.execute_private(request)

    assert called == [private_joint]
    assert event.postcondition is OpenPostcondition.SATISFIED
    assert event.public_result.completed
    assert redact_interaction_event(event) == {
        "request_id": "request_1",
        "instance_id": "obj_000017",
        "operation": "open",
        "status": "completed",
    }
    public_trace = json.dumps(redact_interaction_event(event), sort_keys=True)
    assert private_joint.source_object_name not in public_trace
    assert private_joint.joint_name not in public_trace
    assert str(private_joint.joint_index) not in public_trace


def test_open_requires_observable_after_state_not_just_executor_success() -> None:
    registry = OpaqueObjectRegistry()
    registry.register("obj_000001", joints=[object()])
    skill = TrustedInteractionSkill(
        registry,
        lambda _joint: JointOpenResult(True, 0.0, None),
    )

    event = skill.execute_private(ObjectInteractionRequest("request_2", "obj_000001"))

    assert event.postcondition is OpenPostcondition.UNOBSERVABLE
    assert event.public_result.status == "failed"
    assert redact_interaction_event(event)["status"] == "failed"


def test_open_postcondition_can_require_all_private_object_joints() -> None:
    first, second = object(), object()
    registry = OpaqueObjectRegistry()
    registry.register(
        "obj_000003",
        joints=[first, second],
        open_postcondition=OpenPostconditionSpec(success_fraction=0.8, minimum_open_joints=2),
    )

    def execute_open_joint(joint: object) -> JointOpenResult:
        return JointOpenResult(True, 0.0, 1.0 if joint is first else 0.4)

    skill = TrustedInteractionSkill(registry, execute_open_joint)
    event = skill.execute_private(ObjectInteractionRequest("request_3", "obj_000003"))

    assert event.postcondition is OpenPostcondition.NOT_SATISFIED
    assert event.public_result.status == "failed"


def test_preopened_object_is_completed_without_exposing_joint_state() -> None:
    registry = OpaqueObjectRegistry()
    registry.register("obj_000004", joints=[object()])
    skill = TrustedInteractionSkill(
        registry,
        lambda _joint: JointOpenResult(False, 0.9, 0.9),
    )

    event = skill.execute_private(ObjectInteractionRequest("request_4", "obj_000004"))

    assert event.postcondition is OpenPostcondition.ALREADY_SATISFIED
    assert event.public_result.status == "completed"
    assert "open_fraction_after" not in redact_interaction_event(event)


def test_unknown_or_unsupported_requests_are_rejected_without_details() -> None:
    skill = TrustedInteractionSkill(OpaqueObjectRegistry(), lambda _joint: JointOpenResult(True, 0.0, 1.0))

    unknown = skill.execute_private(ObjectInteractionRequest("request_5", "obj_999999"))
    unsupported = skill.execute_private(ObjectInteractionRequest("request_6", "obj_999999", operation="close"))  # type: ignore[arg-type]

    assert unknown.public_result.status == "rejected"
    assert unsupported.public_result.status == "rejected"
    assert set(redact_interaction_event(unknown)) == {"request_id", "instance_id", "operation", "status"}


def test_public_request_parser_rejects_private_selector_fields() -> None:
    with pytest.raises(ValueError, match="unsupported public interaction fields"):
        ObjectInteractionRequest.from_public_payload(
            {
                "request_id": "request_7",
                "instance_id": "obj_000007",
                "operation": "open",
                "joint_name": "private_hinge_joint",
            }
        )


def test_public_request_parser_requires_opaque_string_ids() -> None:
    with pytest.raises(TypeError, match="instance_id must be a string"):
        ObjectInteractionRequest.from_public_payload(
            {
                "request_id": "request_8",
                "instance_id": 8,
                "operation": "open",
            }
        )


def test_postcondition_classification_does_not_use_intermediate_control_steps() -> None:
    postcondition = classify_open_postcondition(
        [JointOpenResult(True, 0.1, 0.8, metadata={"internal_steps": 500})],
        spec=OpenPostconditionSpec(success_fraction=0.8),
    )

    assert postcondition is OpenPostcondition.SATISFIED
