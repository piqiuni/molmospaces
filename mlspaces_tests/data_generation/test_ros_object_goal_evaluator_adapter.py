"""Protocol tests for the evaluator-owned restricted-GT ROS adapter.

No ROS, MuJoCo task, external semantic-mapping package, or force controller is
needed here.  The tests assert the evaluator boundary before it is wired into a
live V3 rollout loop.
"""

from __future__ import annotations

import json

import pytest

from scripts.InteractiveNav.evaluation.benchmark_interaction_adapter import (
    validate_public_interaction_pose,
)
from scripts.InteractiveNav.evaluation.ros_object_goal_adapter import (
    DIRECT_DRAWER_SCAN_BBOX_TTL_S,
    RestrictedGTContractError,
    RestrictedGTObservation,
    RosObjectGoalEvaluatorAdapter,
    build_public_target_context,
    validate_semantic_minimal_perception_payload,
)


class _FakeString:
    def __init__(self, *, data: str) -> None:
        self.data = data


class _FakePublisher:
    def __init__(self, topic: str, **_kwargs) -> None:
        self.topic = topic
        self.messages: list[_FakeString] = []

    def publish(self, message: _FakeString) -> None:
        self.messages.append(message)


class _FakeSubscriber:
    def __init__(self, topic: str, callback, **_kwargs) -> None:
        self.topic = topic
        self.callback = callback
        self.unregistered = False

    def unregister(self) -> None:
        self.unregistered = True


class _FakeRospy:
    def __init__(self) -> None:
        self.publishers: dict[str, _FakePublisher] = {}
        self.subscribers: dict[str, _FakeSubscriber] = {}

    def Publisher(self, topic: str, _message_type, **kwargs) -> _FakePublisher:
        publisher = _FakePublisher(topic, **kwargs)
        self.publishers[topic] = publisher
        return publisher

    def Subscriber(self, topic: str, _message_type, callback, **kwargs) -> _FakeSubscriber:
        subscriber = _FakeSubscriber(topic, callback, **kwargs)
        self.subscribers[topic] = subscriber
        return subscriber


def _adapter(
    *,
    executor=None,
    clock=None,
    require_public_interaction_evidence: bool = False,
    queue_rejected_interactions: bool = False,
) -> tuple[RosObjectGoalEvaluatorAdapter, _FakeRospy]:
    rospy = _FakeRospy()
    adapter = RosObjectGoalEvaluatorAdapter(
        rospy_module=rospy,
        string_message_type=_FakeString,
        interaction_executor=executor,
        require_public_interaction_evidence=require_public_interaction_evidence,
        queue_rejected_interactions=queue_rejected_interactions,
        clock=clock or (lambda: 123.5),
    )
    return adapter, rospy


def _reset(adapter: RosObjectGoalEvaluatorAdapter, private_instances: dict[str, object]) -> None:
    adapter.reset(
        episode_id="eval_000042",
        target_context=build_public_target_context(
            episode_id="eval_000042",
            target_name="refrigerator",
            object_labels=["fridge", "refrigerator"],
            instruction="Find the refrigerator.",
        ),
        private_instances=private_instances,
    )


def _payload(message: _FakeString) -> dict:
    return json.loads(message.data)


def test_restricted_payload_is_compact_semantic_minimal_gt_without_private_fields() -> None:
    private_handle = {"source_object_name": "raw_fridge_body_928", "joint_name": "hinge_928"}
    adapter, rospy = _adapter()
    _reset(adapter, {"obj_000017": private_handle})

    payload = adapter.publish_observations(
        [
            RestrictedGTObservation(
                instance_id="obj_000017",
                name="refrigerator",
                bbox_2d_xyxy=[0, 0, 1, 1],
                segmentation_rle={"size": [2, 2], "counts": [0, 4]},
                box3d_center=[1.0, 2.0, 0.5],
                box3d_size=[0.8, 0.6, 1.7],
            )
        ],
        capture_step=3,
    )

    validate_semantic_minimal_perception_payload(payload)
    published = _payload(rospy.publishers[adapter.gt_observations_topic].messages[-1])
    observation = published["observations"][0]
    assert set(observation) == {"id", "name", "bbox_2d", "mask_rle", "box_3d"}
    assert observation == {
        "id": "obj_000017",
        "name": "refrigerator",
        "bbox_2d": [0.0, 0.0, 1.0, 1.0],
        "mask_rle": {"size": [2, 2], "counts": [0, 4]},
        "box_3d": {
            "center": [1.0, 2.0, 0.5],
            "size": [0.8, 0.6, 1.7],
            "frame_id": "world",
        },
    }
    serialized = json.dumps(published, sort_keys=True)
    assert "raw_fridge_body_928" not in serialized
    assert "hinge_928" not in serialized
    assert "joint_infos" not in serialized
    assert "orientation" not in serialized
    assert "source_object_name" not in serialized
    assert "visible_pixels" not in serialized


def test_v3_restricted_frame_keeps_mask_rle_compact_on_semantic_wire() -> None:
    private_handle = {"source_object_name": "private_door_root", "joint_name": "private_hinge"}
    adapter, rospy = _adapter()
    adapter.reset(
        episode_id="episode_000001",
        target_context=build_public_target_context(
            episode_id="episode_000001",
            target_name="chair",
        ),
        private_instances={"obj_000003": private_handle},
    )
    frame = {
        "protocol_version": "interactive_nav_v3_restricted_gt_v1",
        "episode_id": "episode_000001",
        "episode_reset": False,
        "frame_index": 7,
        "observations": [
            {
                "instance_id": "obj_000003",
                "name": "door",
                "bbox_2d_xyxy": [1, 1, 2, 2],
                "mask_rle": {"size": [4, 4], "counts": [5, 2, 2, 2, 5]},
                "bbox_3d": {
                    "center": [2.0, 1.0, 1.0],
                    "size": [0.2, 1.0, 2.0],
                    "frame_id": "world",
                },
            }
        ],
    }

    payload = adapter.publish_restricted_gt_frame(frame, capture_step=11, stamp_sec=42.0)

    validate_semantic_minimal_perception_payload(payload)
    assert payload["capture_step"] == 11
    assert payload["stamp_sec"] == 42.0
    observation = _payload(rospy.publishers[adapter.gt_observations_topic].messages[-1])["observations"][0]
    assert observation == {
        "id": "obj_000003",
        "name": "door",
        "bbox_2d": [1, 1, 2, 2],
        "mask_rle": {"size": [4, 4], "counts": [5, 2, 2, 2, 5]},
        "box_3d": {
            "center": [2.0, 1.0, 1.0],
            "size": [0.2, 1.0, 2.0],
            "frame_id": "world",
        },
    }
    serialized = json.dumps(payload, sort_keys=True)
    assert "private_door_root" not in serialized
    assert "private_hinge" not in serialized
    assert "source_object_name" not in serialized
    assert "visible_pixels" not in serialized


def test_object_level_command_uses_private_handle_but_redacts_force_result() -> None:
    private_handle = object()
    adapter, rospy = _adapter()
    _reset(adapter, {"obj_000017": private_handle})

    request = adapter.receive_interaction_command(
        {
            "command_id": "decision:open-fridge",
            "node_id": "container_obj_000017",
            "object_id": "obj_000017",
            "action": "open",
            # A legacy method may send these fields.  The adapter must not pass
            # them into its trusted object-level force-skill API.
            "joint_names": ["guessed_hinge"],
        }
    )

    assert request is not None
    assert request.private_handle is private_handle
    assert request.instance_id == "obj_000017"
    result = adapter.complete_interaction(request.command_id, success=True)
    assert result["success"] is True
    assert result["status"] == "SUCCEEDED"
    assert result["object_id"] == "obj_000017"
    assert result["instance_id"] == "obj_000017"
    assert result["state"] == "open"
    assert result["post_state"] == "open"
    assert result["interaction_capability"] == "articulated"
    assert result["interactable"] is True
    assert result["verification_source"] == "executor_state_verification"
    assert "source_object_name" not in result
    assert "joint_names" not in result
    assert "joint_infos" not in result
    assert "private_handle" not in result
    published = _payload(rospy.publishers[adapter.interaction_result_topic].messages[-1])
    assert published == result


def test_public_portal_alias_resolves_to_the_canonical_opaque_id() -> None:
    """A generic public door token must not lose its evaluator skill route."""

    private_handle = object()
    adapter, _rospy = _adapter()
    adapter.reset(
        episode_id="eval_000042",
        target_context=build_public_target_context(
            episode_id="eval_000042",
            target_name="door",
        ),
        private_instances={"obj_000017": private_handle},
        instance_aliases={"door_7031": "obj_000017"},
    )

    request = adapter.receive_interaction_command(
        {
            "command_id": "decision:open-generic-door",
            "node_id": "door_7031",
            "object_id": "door_7031",
            "node_type": "portal",
            "action": "open",
        }
    )

    assert request is not None
    assert request.private_handle is private_handle
    assert request.instance_id == "obj_000017"


def test_unpublished_opaque_ids_do_not_reveal_private_registration() -> None:
    adapter, rospy = _adapter(
        require_public_interaction_evidence=True,
        queue_rejected_interactions=True,
    )
    adapter.reset(
        episode_id="eval_000042",
        target_context=build_public_target_context(
            episode_id="eval_000042",
            target_name="refrigerator",
            object_labels=["fridge", "refrigerator"],
            instruction="Find the refrigerator.",
        ),
        private_instances={"obj_000017": object()},
        instance_aliases={"door_7031": "obj_000017"},
    )

    request = adapter.receive_interaction_command(
        {
            "command_id": "guessed-before-public-frame",
            "object_id": "obj_000017",
            "action": "open",
        }
    )

    assert request is not None
    assert request.instance_id == "obj_000017"
    assert request.private_handle is None
    assert request.public_observation == {}
    assert request.rejection_reason == "interaction_not_visible"
    assert adapter.pending_interaction_count == 1
    assert rospy.publishers[adapter.interaction_result_topic].messages == []
    assert adapter.pop_next_interaction_request() is request

    unknown = adapter.receive_interaction_command(
        {
            "command_id": "guessed-unknown-before-public-frame",
            "object_id": "obj_999999",
            "action": "open",
        }
    )
    assert unknown is not None
    assert unknown.instance_id == "obj_999999"
    assert unknown.private_handle is None
    assert unknown.public_observation == {}
    assert unknown.rejection_reason == request.rejection_reason

    guessed_alias = adapter.receive_interaction_command(
        {
            "command_id": "guessed-alias-before-public-frame",
            "object_id": "door_7031",
            "node_type": "portal",
            "action": "open",
        }
    )
    assert guessed_alias is not None
    assert guessed_alias.instance_id == "door_7031"
    assert guessed_alias.private_handle is None
    assert guessed_alias.rejection_reason == "interaction_not_visible"


def test_recent_published_box_3d_is_attached_as_public_interaction_provenance() -> None:
    now = [100.0]
    private_handle = object()
    adapter, _rospy = _adapter(
        clock=lambda: now[0],
        require_public_interaction_evidence=True,
        queue_rejected_interactions=True,
    )
    _reset(adapter, {"obj_000017": private_handle})
    adapter.publish_observations(
        [
            RestrictedGTObservation(
                instance_id="obj_000017",
                name="refrigerator",
                bbox_2d_xyxy=[0, 0, 1, 1],
                segmentation_rle={"size": [2, 2], "counts": [0, 4]},
                box3d_center=[1.0, 2.0, 0.5],
                box3d_size=[0.8, 0.6, 1.7],
            )
        ],
        capture_step=12,
    )

    request = adapter.receive_interaction_command(
        {
            "command_id": "open-from-public-box",
            "object_id": "obj_000017",
            "action": "open",
        }
    )

    assert request is not None
    assert request.private_handle is private_handle
    assert request.rejection_reason == ""
    assert request.public_observation == {
        "capture_step": 12,
        "age_seconds": 0.0,
        "box_3d": {
            "center": [1.0, 2.0, 0.5],
            "size": [0.8, 0.6, 1.7],
            "frame_id": "world",
        },
    }


def test_expired_public_box_3d_is_queued_as_not_visible() -> None:
    now = [100.0]
    adapter, _rospy = _adapter(
        clock=lambda: now[0],
        require_public_interaction_evidence=True,
        queue_rejected_interactions=True,
    )
    _reset(adapter, {"obj_000017": object()})
    adapter.publish_observations(
        [
            RestrictedGTObservation(
                instance_id="obj_000017",
                name="refrigerator",
                bbox_2d_xyxy=[0, 0, 1, 1],
                segmentation_rle={"size": [2, 2], "counts": [0, 4]},
                box3d_center=[1.0, 2.0, 0.5],
                box3d_size=[0.8, 0.6, 1.7],
            )
        ],
        capture_step=12,
    )
    now[0] += DIRECT_DRAWER_SCAN_BBOX_TTL_S + 0.01

    request = adapter.receive_interaction_command(
        {
            "command_id": "open-after-public-box-expired",
            "object_id": "obj_000017",
            "action": "open",
        }
    )

    assert request is not None
    assert request.rejection_reason == "interaction_not_visible"
    assert request.public_observation == {}
    assert adapter.pending_interaction_count == 1
    assert adapter.pop_next_interaction_request() is request


@pytest.mark.parametrize(
    ("payload", "expected_reason"),
    [
        (
            {
                "command_id": "unknown-nonportal",
                "object_id": "obj_not_registered",
                "node_type": "container",
                "action": "open",
            },
            "unknown_instance_id",
        ),
        (
            {
                "command_id": "unsupported-close",
                "object_id": "obj_000017",
                "node_type": "container",
                "action": "close",
            },
            "unsupported_action",
        ),
    ],
)
def test_unknown_nonportal_and_unsupported_commands_are_queued_for_accounting(
    payload: dict,
    expected_reason: str,
) -> None:
    adapter, rospy = _adapter(queue_rejected_interactions=True)
    _reset(adapter, {"obj_000017": object()})

    request = adapter.receive_interaction_command(payload)

    assert request is not None
    assert request.rejection_reason == expected_reason
    assert adapter.pending_interaction_count == 1
    assert rospy.publishers[adapter.interaction_result_topic].messages == []
    assert adapter.pop_next_interaction_request() is request

    result = adapter.complete_interaction(
        request.command_id,
        success=False,
        status="INVALID",
        reason=request.rejection_reason,
    )
    assert result["success"] is False
    assert adapter.pending_interaction_count == 0
    assert len(adapter.published_result_events) == 1


def test_drawer_scan_visual_hint_is_sanitized_but_never_emitted_in_result() -> None:
    adapter, rospy = _adapter()
    _reset(adapter, {"obj_000017": object()})

    request = adapter.receive_interaction_command(
        {
            "command_id": "scan-dresser",
            "object_id": "obj_000017",
            "action": "scan",
            "sequence_type": "drawer_scan",
            "open_regions": [
                {"center": [0.6, 0.8], "confidence": 0.9},
                {"center": [0.4, 0.2], "confidence": 0.8},
                {"center": [4.0, 0.5]},
                {"center": [0.4, 0.2]},
            ],
            "joint_names": ["guessed_private_drawer"],
            "force_target_fraction": 1.0,
        }
    )

    assert request is not None
    assert request.action == "scan"
    assert request.sequence_type == "drawer_scan"
    assert request.open_regions == ((0.4, 0.2), (0.6, 0.8))
    assert request.public_command["open_regions"] == [
        {"center": [0.6, 0.8]},
        {"center": [0.4, 0.2]},
    ]
    assert "joint_names" not in request.public_command
    assert "force_target_fraction" not in request.public_command
    result = adapter.complete_interaction(request.command_id, success=True)
    assert result["action"] == "scan"
    serialized = json.dumps(result, sort_keys=True)
    assert "drawer_scan" not in serialized
    assert "guessed_private_drawer" not in serialized
    assert "force_target_fraction" not in serialized
    assert _payload(rospy.publishers[adapter.interaction_result_topic].messages[-1]) == result


def test_drawer_open_uses_open_action_and_retains_visual_grounding() -> None:
    adapter, rospy = _adapter()
    _reset(adapter, {"obj_000017": object()})

    request = adapter.receive_interaction_command(
        {
            "command_id": "open-dresser-drawer",
            "object_id": "obj_000017",
            "action": "open",
            "sequence_type": "drawer_open",
            "open_regions": [{"center": [0.25, 0.75]}],
        }
    )

    assert request is not None
    assert request.action == "open"
    assert request.sequence_type == "drawer_open"
    assert request.open_regions == ((0.25, 0.75),)
    result = adapter.complete_interaction(
        request.command_id,
        success=True,
        outcome={
            "state": "open",
            "post_state": "open",
            "verification_source": "executor_state_verification",
        },
    )
    assert result["action"] == "open"
    assert result["state"] == "open"
    assert _payload(rospy.publishers[adapter.interaction_result_topic].messages[-1]) == result


def test_direct_bbox_drawer_scan_routes_a_unique_current_public_box() -> None:
    private_handle = object()
    adapter, rospy = _adapter()
    _reset(adapter, {"obj_000017": private_handle, "obj_000018": object()})
    adapter.publish_observations(
        [
            RestrictedGTObservation(
                instance_id="obj_000017",
                name="dresser",
                bbox_2d_xyxy=[0, 0, 1, 1],
                segmentation_rle={"size": [4, 4], "counts": [0, 16]},
                box3d_center=[1.0, 2.0, 0.5],
                box3d_size=[0.8, 0.6, 1.7],
            ),
            RestrictedGTObservation(
                instance_id="obj_000018",
                name="cabinet",
                bbox_2d_xyxy=[2, 0, 3, 1],
                segmentation_rle={"size": [4, 4], "counts": [0, 16]},
                box3d_center=[2.0, 2.0, 0.5],
                box3d_size=[0.8, 0.6, 1.7],
            ),
        ],
        capture_step=3,
    )

    request = adapter.receive_interaction_command(
        {
            "command_id": "scan-visible-dresser",
            # Direct scan identity comes from the public box, not this stale
            # method-side selector.
            "object_id": "not-an-evaluator-id",
            "action": "open",
            "sequence_type": "drawer_scan",
            "drawer_container_bbox_2d": [0, 0, 1, 1],
            "drawer_container_capture_step": 3,
        }
    )

    assert request is not None
    assert request.private_handle is private_handle
    assert request.instance_id == "obj_000017"
    assert request.sequence_type == "drawer_scan"
    assert request.open_regions == ()
    assert request.direct_bbox_drawer_scan is True
    result = adapter.complete_interaction(request.command_id, success=True)
    serialized = json.dumps(result, sort_keys=True)
    assert "drawer_container_bbox_2d" not in serialized
    assert "direct_bbox_drawer_scan" not in serialized
    assert "not-an-evaluator-id" not in serialized
    assert _payload(rospy.publishers[adapter.interaction_result_topic].messages[-1]) == result


def test_direct_bbox_drawer_scan_routes_exact_public_box_after_approach_lag() -> None:
    """A graph-selected drawer box may arrive after its approach navigation."""

    now = [100.0]
    private_handle = object()
    adapter, rospy = _adapter(clock=lambda: now[0])
    _reset(adapter, {"obj_000017": private_handle})
    drawer = RestrictedGTObservation(
        instance_id="obj_000017",
        name="dresser",
        bbox_2d_xyxy=[0, 0, 1, 1],
        segmentation_rle={"size": [4, 4], "counts": [0, 16]},
        box3d_center=[1.0, 2.0, 0.5],
        box3d_size=[0.8, 0.6, 1.7],
    )
    adapter.publish_observations([drawer], capture_step=455)
    # The evaluator keeps publishing during navigation, but the semantic graph
    # can still carry the last public drawer crop when the executor is ready.
    # This mirrors the observed 455 -> 532 V3 delay.
    for capture_step in range(456, 533):
        adapter.publish_observations([], capture_step=capture_step)
    now[0] += DIRECT_DRAWER_SCAN_BBOX_TTL_S * 0.75

    request = adapter.receive_interaction_command(
        {
            "command_id": "scan-after-approach",
            "object_id": "stale-method-selector",
            "action": "open",
            "sequence_type": "drawer_scan",
            "drawer_container_bbox_2d": [0, 0, 1, 1],
            "drawer_container_capture_step": 455,
        }
    )

    assert request is not None
    assert request.private_handle is private_handle
    assert request.instance_id == "obj_000017"
    assert request.sequence_type == "drawer_scan"
    assert request.direct_bbox_drawer_scan is True
    result = adapter.complete_interaction(request.command_id, success=True)
    serialized = json.dumps(result, sort_keys=True)
    assert "stale-method-selector" not in serialized
    assert "drawer_container_bbox_2d" not in serialized


def test_direct_bbox_drawer_scan_rejects_noncurrent_or_ambiguous_public_boxes() -> None:
    adapter, rospy = _adapter()
    _reset(adapter, {"obj_000017": object(), "obj_000018": object()})
    observation = RestrictedGTObservation(
        instance_id="obj_000017",
        name="dresser",
        bbox_2d_xyxy=[0, 0, 1, 1],
        segmentation_rle={"size": [4, 4], "counts": [0, 16]},
        box3d_center=[1.0, 2.0, 0.5],
        box3d_size=[0.8, 0.6, 1.7],
    )
    adapter.publish_observations([observation], capture_step=3)
    # The requested public step is empty, so the box cannot select the older
    # frame even though the adapter retains a small asynchronous history.
    adapter.publish_observations([], capture_step=4)
    assert adapter.receive_interaction_command(
        {
            "command_id": "scan-stale-dresser",
            "action": "open",
            "sequence_type": "drawer_scan",
            "drawer_container_bbox_2d": [0, 0, 1, 1],
            "drawer_container_capture_step": 4,
        }
    ) is None
    stale = _payload(rospy.publishers[adapter.interaction_result_topic].messages[-1])
    assert stale["status"] == "REJECTED"
    assert stale["reason"] == "unresolved_drawer_scan_target"
    assert stale["object_id"] == ""
    assert stale["state"] == "unavailable"
    assert stale["interaction_capability"] == "unavailable"
    assert stale["interactable"] is False
    assert stale["retryable"] is False
    assert stale["verification_source"] == "executor_capability_check"

    # Matching must also be unique; a box cannot select among two public
    # instances that overlap equally.
    adapter.publish_observations(
        [
            observation,
            RestrictedGTObservation(
                instance_id="obj_000018",
                name="cabinet",
                bbox_2d_xyxy=[0, 0, 1, 1],
                segmentation_rle={"size": [4, 4], "counts": [0, 16]},
                box3d_center=[2.0, 2.0, 0.5],
                box3d_size=[0.8, 0.6, 1.7],
            ),
        ],
        capture_step=5,
    )
    assert adapter.receive_interaction_command(
        {
            "command_id": "scan-ambiguous-dresser",
            "action": "open",
            "sequence_type": "drawer_scan",
            "drawer_container_bbox_2d": [0, 0, 1, 1],
            "drawer_container_capture_step": 5,
        }
    ) is None
    ambiguous = _payload(rospy.publishers[adapter.interaction_result_topic].messages[-1])
    assert ambiguous["status"] == "REJECTED"
    assert ambiguous["reason"] == "unresolved_drawer_scan_target"
    assert "obj_000017" not in json.dumps(ambiguous, sort_keys=True)
    assert "obj_000018" not in json.dumps(ambiguous, sort_keys=True)


def test_direct_bbox_drawer_scan_requires_matching_public_step_and_fresh_frame() -> None:
    now = [100.0]
    adapter, rospy = _adapter(clock=lambda: now[0])
    _reset(adapter, {"obj_000017": object()})
    observation = RestrictedGTObservation(
        instance_id="obj_000017",
        name="dresser",
        bbox_2d_xyxy=[0, 0, 1, 1],
        segmentation_rle={"size": [4, 4], "counts": [0, 16]},
        box3d_center=[1.0, 2.0, 0.5],
        box3d_size=[0.8, 0.6, 1.7],
    )
    adapter.publish_observations([observation], capture_step=7)

    def rejected(command_id: str, **hint) -> None:
        assert adapter.receive_interaction_command(
            {
                "command_id": command_id,
                # A valid opaque selector must not turn an invalid direct-bbox
                # request into an ordinary MLLM drawer scan.
                "object_id": "obj_000017",
                "action": "open",
                "sequence_type": "drawer_scan",
                "drawer_container_bbox_2d": [0, 0, 1, 1],
                **hint,
            }
        ) is None
        result = _payload(rospy.publishers[adapter.interaction_result_topic].messages[-1])
        assert result["status"] == "REJECTED"
        assert result["reason"] == "unresolved_drawer_scan_target"

    # A box from the public frame cannot be rebound to an arbitrary older or
    # future frame token.
    rejected("scan-old-step", drawer_container_capture_step=6)
    rejected("scan-negative-step", drawer_container_capture_step=-1)
    # Direct bbox routing requires the public step field; a normal MLLM scan
    # remains compatible because it sends no drawer_container_bbox_2d field.
    rejected("scan-missing-step")

    now[0] += DIRECT_DRAWER_SCAN_BBOX_TTL_S + 0.01
    rejected("scan-expired-frame", drawer_container_capture_step=7)


def test_episode_reset_clears_direct_bbox_frame_history() -> None:
    adapter, rospy = _adapter()
    _reset(adapter, {"obj_000017": object()})
    adapter.publish_observations(
        [
            RestrictedGTObservation(
                instance_id="obj_000017",
                name="dresser",
                bbox_2d_xyxy=[0, 0, 1, 1],
                segmentation_rle={"size": [4, 4], "counts": [0, 16]},
                box3d_center=[1.0, 2.0, 0.5],
                box3d_size=[0.8, 0.6, 1.7],
            )
        ],
        capture_step=7,
    )

    # Keep the public episode ID intentionally unchanged: generation, not
    # merely a string comparison, must prevent cache reuse across a reset.
    _reset(adapter, {"obj_000018": object()})
    assert adapter.receive_interaction_command(
        {
            "command_id": "scan-before-reset",
            "object_id": "obj_000018",
            "action": "open",
            "sequence_type": "drawer_scan",
            "drawer_container_bbox_2d": [0, 0, 1, 1],
            "drawer_container_capture_step": 7,
        }
    ) is None
    result = _payload(rospy.publishers[adapter.interaction_result_topic].messages[-1])
    assert result["status"] == "REJECTED"
    assert result["reason"] == "unresolved_drawer_scan_target"


def test_executor_receives_private_handle_and_unknown_raw_name_is_rejected() -> None:
    private_handle = {"body": "private_mujoco_name", "joint": "private_joint"}
    received = []

    def executor(request):
        received.append(request)
        return {"success": True, "joint_infos": [{"joint_name": "private_joint"}]}

    adapter, rospy = _adapter(executor=executor)
    _reset(adapter, {"obj_000017": private_handle})
    assert adapter.receive_interaction_command(
        {
            "command_id": "open-1",
            "source_object_name": "obj_000017",
            "action": "open",
        }
    ) is None
    assert len(received) == 1
    assert received[0].private_handle is private_handle
    success_result = _payload(rospy.publishers[adapter.interaction_result_topic].messages[-1])
    assert success_result["success"] is True
    assert "private_joint" not in json.dumps(success_result)
    assert "private_mujoco_name" not in json.dumps(success_result)

    assert adapter.receive_interaction_command(
        {
            "command_id": "open-raw-name",
            "source_object_name": "private_mujoco_name",
            "action": "open",
        }
    ) is None
    rejected = _payload(rospy.publishers[adapter.interaction_result_topic].messages[-1])
    assert rejected["status"] == "REJECTED"
    assert rejected["success"] is False
    assert rejected["reason"] == "unknown_instance_id"
    assert len(received) == 1


def test_unknown_semantic_portal_is_unavailable_without_public_aperture_evidence() -> None:
    adapter, rospy = _adapter()
    _reset(adapter, {"obj_000017": object()})

    request = adapter.receive_interaction_command(
        {
            "command_id": "open-static-doorway",
            "node_id": "portal_obj_000021",
            "candidate_id": "interaction:portal_obj_000021:open",
            "node_type": "portal",
            "object_id": "obj_000021",
            "action": "open",
        }
    )

    assert request is not None
    assert request.private_handle is None
    assert request.rejection_reason == "unknown_instance_id"
    assert adapter.pending_interaction_count == 1
    # Production evaluator code consumes this queued marker and emits the
    # result with INVALID status in the same decision turn.
    result = adapter.complete_interaction(
        request.command_id,
        success=False,
        status="INVALID",
        reason=request.rejection_reason,
    )
    assert result["status"] == "FAILED"
    assert result["success"] is False
    assert result["reason"] == "capability_unavailable"
    assert result["state"] == "unavailable"
    assert result["interactable"] is False
    assert result["retryable"] is False
    assert result["interaction_capability"] == "unavailable"
    assert _payload(rospy.publishers[adapter.interaction_result_topic].messages[-1]) == result


def test_unknown_semantic_portal_requires_two_factor_public_static_open_evidence() -> None:
    adapter, rospy = _adapter()
    _reset(adapter, {"obj_000017": object()})

    request = adapter.receive_interaction_command(
        {
            "command_id": "open-fixed-aperture",
            "node_id": "portal_obj_000021",
            "candidate_id": "interaction:portal_obj_000021:open",
            "node_type": "portal",
            "object_id": "obj_000021",
            "action": "open",
            "portal_aperture_observation": {
                "door_leaf": "absent",
                "connectivity": "open",
                "confidence": 0.9,
                "private_body_name": "must_not_escape",
            },
        }
    )

    assert request is not None
    result = adapter.complete_interaction(
        request.command_id,
        success=False,
        status="INVALID",
        reason=request.rejection_reason,
    )
    assert result["status"] == "SUCCEEDED"
    assert result["success"] is True
    assert result["state"] == "static_open"
    assert result["interaction_capability"] == "static"
    assert result["portal_aperture_observation"] == {
        "door_leaf": "absent",
        "connectivity": "open",
        "confidence": 0.9,
    }
    assert "must_not_escape" not in json.dumps(result)
    assert _payload(rospy.publishers[adapter.interaction_result_topic].messages[-1]) == result


def test_rich_backend_outcome_is_allowlisted_and_redacts_private_force_fields() -> None:
    adapter, rospy = _adapter()
    _reset(adapter, {"obj_000017": object()})
    request = adapter.receive_interaction_command(
        {
            "command_id": "open-rich-result",
            "node_id": "container_obj_000017",
            "object_id": "obj_000017",
            "node_type": "container",
            "action": "open",
            "interaction_approach_pose_xyyaw": [1.0, 2.0, 0.3],
            "joint_names": ["guessed_private_joint"],
        }
    )
    assert request is not None

    result = adapter.complete_interaction(
        request.command_id,
        success=False,
        outcome={
            "state": "blocked",
            "post_state": "blocked",
            "interaction_capability": "blocked",
            "interactable": False,
            "retryable": False,
            "failure_reason": "force_target_not_reached",
            "verification_source": "executor_state_verification",
            "physics_substeps": 123,
            "task_steps_consumed": 4,
            "recommended_retreat_m": 0.25,
            "interaction_pose_validation": {
                "checked": True,
                "valid": True,
                "expected_pose_xyyaw": [1.0, 2.0, 0.3],
                "actual_pose_xyyaw": [1.02, 2.01, 0.31],
                "private_joint_name": "hidden_joint",
            },
            "source_object_name": "hidden_container",
            "joint_infos": [{"joint_name": "hidden_joint"}],
            "final_joint_open_fractions": {"hidden_joint": 0.2},
            "view_profile_result": {"joint_name": "robot_private_head_joint"},
            "oracle_interaction_id": "private_recipe_step",
        },
    )

    assert result["status"] == "FAILED"
    assert result["state"] == "blocked"
    assert result["interaction_capability"] == "blocked"
    assert result["failure_reason"] == "force_target_not_reached"
    assert result["physics_substeps"] == 123
    assert result["task_steps_consumed"] == 4
    assert result["recommended_retreat_m"] == 0.25
    assert result["interaction_pose_validation"]["valid"] is True
    serialized = json.dumps(result, sort_keys=True)
    for forbidden in (
        "hidden_container",
        "hidden_joint",
        "robot_private_head_joint",
        "private_recipe_step",
        "joint_infos",
        "final_joint_open_fractions",
        "oracle_interaction_id",
    ):
        assert forbidden not in serialized
    assert _payload(rospy.publishers[adapter.interaction_result_topic].messages[-1]) == result


def test_explicit_private_failure_reason_is_normalized_before_publication() -> None:
    adapter, rospy = _adapter()
    _reset(adapter, {"obj_000017": object()})
    request = adapter.receive_interaction_command(
        {
            "command_id": "private-error-redaction",
            "object_id": "obj_000017",
            "action": "open",
        }
    )
    assert request is not None

    result = adapter.complete_interaction(
        request.command_id,
        success=False,
        reason="private_joint_X failed at qpos 0.23",
    )

    assert result["reason"] == "executor_failed"
    assert result["failure_reason"] == "executor_failed"
    assert "private_joint" not in json.dumps(result, sort_keys=True).casefold()
    assert _payload(rospy.publishers[adapter.interaction_result_topic].messages[-1]) == result


def test_public_pose_validation_rejects_wrong_yaw_without_object_oracle() -> None:
    validation = validate_public_interaction_pose(
        {
            "interaction_approach_pose_xyyaw": [1.0, 2.0, 0.0],
            "interaction_ready_distance_m": 0.2,
            "interaction_ready_yaw_tolerance_rad": 0.25,
            "interaction_approach_axis_xy": [1.0, 0.0],
            # Private selectors are irrelevant to this public-pose check.
            "joint_names": ["must_not_be_read"],
        },
        actual_pose_xyyaw=[1.05, 2.0, 1.2],
    )

    assert validation["checked"] is True
    assert validation["valid"] is False
    assert validation["position_error_m"] == pytest.approx(0.05)
    assert validation["yaw_error_rad"] == pytest.approx(1.2)
    assert validation["approach_axis_xy"] == [1.0, 0.0]
    assert "joint" not in json.dumps(validation, sort_keys=True)


def test_public_pose_validation_without_expected_pose_preserves_legacy_access_gate() -> None:
    assert validate_public_interaction_pose(
        {},
        actual_pose_xyyaw=[1.0, 2.0, 0.0],
    ) == {
        "checked": False,
        "reason": "no_expected_approach_pose",
    }


def test_reset_clears_pending_command_and_publishes_empty_episode_marker() -> None:
    adapter, rospy = _adapter()
    _reset(adapter, {"obj_000017": object()})
    request = adapter.receive_interaction_command(
        {"command_id": "will-be-cleared", "source_object_name": "obj_000017", "action": "open"}
    )
    assert request is not None
    assert adapter.pending_interaction_count == 1

    _reset(adapter, {"obj_000018": object()})
    assert adapter.pending_interaction_count == 0
    assert adapter.pop_next_interaction_request() is None
    reset_payload = _payload(rospy.publishers[adapter.gt_observations_topic].messages[-1])
    validate_semantic_minimal_perception_payload(reset_payload)
    assert reset_payload["schema_version"] == "interactive_nav_v3_semantic_minimal_gt_v1"
    assert reset_payload["episode_reset"] is True
    assert reset_payload["observations"] == []


def test_target_context_rejects_private_v3_fields() -> None:
    with pytest.raises(RestrictedGTContractError, match="private field"):
        adapter, _ = _adapter()
        adapter.reset(
            episode_id="eval_000042",
            target_context={
                "schema_version": 1,
                "episode_id": "eval_000042",
                "enabled": True,
                "target_name": "refrigerator",
                "object_labels": ["refrigerator"],
                "instruction": "Find it.",
                "target_instance_id": "private_target",
            },
            private_instances={},
        )
