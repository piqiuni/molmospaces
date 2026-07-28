"""Protocol tests for the evaluator-owned restricted-GT ROS adapter.

No ROS, MuJoCo task, external semantic-mapping package, or force controller is
needed here.  The tests assert the evaluator boundary before it is wired into a
live V3 rollout loop.
"""

from __future__ import annotations

import json

import pytest

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


def _adapter(*, executor=None, clock=None) -> tuple[RosObjectGoalEvaluatorAdapter, _FakeRospy]:
    rospy = _FakeRospy()
    adapter = RosObjectGoalEvaluatorAdapter(
        rospy_module=rospy,
        string_message_type=_FakeString,
        interaction_executor=executor,
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
    assert "state" not in result
    assert "source_object_name" not in result
    assert "joint_names" not in result
    assert "joint_infos" not in result
    assert "private_handle" not in result
    published = _payload(rospy.publishers[adapter.interaction_result_topic].messages[-1])
    assert published == result


def test_drawer_scan_visual_hint_is_sanitized_but_never_emitted_in_result() -> None:
    adapter, rospy = _adapter()
    _reset(adapter, {"obj_000017": object()})

    request = adapter.receive_interaction_command(
        {
            "command_id": "scan-dresser",
            "object_id": "obj_000017",
            "action": "open",
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
    assert request.action == "open"
    assert request.sequence_type == "drawer_scan"
    assert request.open_regions == ((0.4, 0.2), (0.6, 0.8))
    result = adapter.complete_interaction(request.command_id, success=True)
    serialized = json.dumps(result, sort_keys=True)
    assert "drawer_scan" not in serialized
    assert "guessed_private_drawer" not in serialized
    assert "force_target_fraction" not in serialized
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
