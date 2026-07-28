import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rospy")

from interaction_attribute_inference_node import InteractionAttributeInferenceNode


class RecordingQueue:
    def __init__(self) -> None:
        self.discarded = []

    def discard(self, object_id, request_sequence=None) -> None:
        self.discarded.append((object_id, request_sequence))


class RecordingPublisher:
    def __init__(self) -> None:
        self.payloads = []

    def publish(self, message) -> None:
        self.payloads.append(json.loads(message.data))


def lifecycle_node() -> InteractionAttributeInferenceNode:
    node = object.__new__(InteractionAttributeInferenceNode)
    node.lock = threading.Lock()
    node.current_episode_id = "episode_1"
    node.episode_active = True
    node.episode_generation = 1
    node.pending_detection_payload = {"observations": [{"id": "door"}]}
    node.pending = {
        "door": {
            "request_sequence": 1,
            "generation": 0,
            "episode_id": "episode_1",
            "episode_generation": 1,
        }
    }
    node.generations = {"door": 0}
    node.last_request = {"door": 1.0}
    node.completed = {"door": {"signature": "door|door"}}
    node.aliases = {"door": "door"}
    node.request_queue = RecordingQueue()
    node.request_timeout_s = 1.0
    node.max_output_tokens = 64
    node._publish_status = lambda: None
    return node


def test_attribute_updates_include_episode_generation() -> None:
    node = lifecycle_node()
    node.publisher = RecordingPublisher()

    node._publish_updates(
        "episode_1",
        2.0,
        [{"object_id": "door", "attribute_status": "ready"}],
    )

    assert node.publisher.payloads == [
        {
            "episode_id": "episode_1",
            "episode_generation": 1,
            "stamp_sec": 2.0,
            "updates": [{"object_id": "door", "attribute_status": "ready"}],
        }
    ]


def test_initial_generation_zero_request_is_current() -> None:
    node = object.__new__(InteractionAttributeInferenceNode)
    node.lock = threading.Lock()
    node.current_episode_id = "episode_1"
    node.pending = {
        "gt_000001": {
            "request_sequence": 1,
            "generation": 0,
            "episode_id": "episode_1",
        }
    }
    node.generations = {"gt_000001": 0}

    assert node._is_current_request("gt_000001", "episode_1", 0, 1)


def test_interaction_result_object_id_invalidates_cached_attribute() -> None:
    node = object.__new__(InteractionAttributeInferenceNode)
    node.lock = threading.Lock()
    node.aliases = {"canonical_object": "canonical_object"}
    node.generations = {"canonical_object": 2}
    node.completed = {"canonical_object": {"signature": "door|canonical_object"}}
    node.last_request = {"canonical_object": 1.0}
    node.pending = {"canonical_object": {"request_sequence": 4}}
    node.request_queue = RecordingQueue()

    node._interaction_result_callback(
        SimpleNamespace(data=json.dumps({"object_id": "canonical_object", "success": True}))
    )

    assert node.generations["canonical_object"] == 3
    assert "canonical_object" not in node.completed
    assert "canonical_object" not in node.last_request
    assert "canonical_object" not in node.pending
    assert node.request_queue.discarded == [("canonical_object", None)]


def test_inactive_target_context_clears_pending_and_rejects_request() -> None:
    node = lifecycle_node()

    node._target_context_callback(
        SimpleNamespace(
            data=json.dumps(
                {
                    "episode_id": "episode_1",
                    "episode_active": False,
                    "episode_generation": 1,
                }
            )
        )
    )

    assert node.episode_active is False
    assert node.pending_detection_payload is None
    assert node.pending == {}
    assert node.request_queue.discarded == [("door", 1)]
    assert not node._is_current_request("door", "episode_1", 0, 1, 1)


def test_inactive_node_does_not_accept_detection_payload() -> None:
    node = lifecycle_node()
    node.episode_active = False
    node.pending_detection_payload = None
    node.filter_counts = {"messages_received": 0}

    node._detection_callback(
        SimpleNamespace(data=json.dumps({"observations": [{"id": "door"}]}))
    )

    assert node.filter_counts["messages_received"] == 1
    assert node.pending_detection_payload is None


@pytest.mark.parametrize(
    ("transition", "expected_active", "expected_generation"),
    [
        ({"episode_active": False, "episode_generation": 1}, False, 1),
        ({"episode_active": True, "episode_generation": 2}, True, 2),
    ],
)
def test_inflight_response_is_stale_across_lifecycle_transition(
    transition, expected_active, expected_generation
) -> None:
    node = lifecycle_node()
    node.publisher = RecordingPublisher()
    node.filter_counts = {
        "started": 0,
        "stale": 0,
        "completed": 0,
        "failed": 0,
    }
    node._encode_jpeg = lambda _crop: b"jpeg"

    class TransitioningClient:
        config = SimpleNamespace(model="test-model")

        def request_json(self, **_kwargs):
            node._target_context_callback(
                SimpleNamespace(data=json.dumps({"episode_id": "episode_2", **transition}))
            )
            return SimpleNamespace(
                error="",
                payload={
                    "object_id": "door",
                    "interactable": True,
                    "interaction_class": "portal",
                    "coarse_state": "closed",
                    "interaction_parts": [],
                    "confidence": 0.9,
                },
            )

    node.client = TransitioningClient()
    node._infer(
        object_id="door",
        detection={"name": "door"},
        crop=np.zeros((2, 2, 3), dtype=np.uint8),
        episode_id="episode_1",
        frame_id="4",
        stamp=1.0,
        signature="door|door",
        generation=0,
        episode_generation=1,
        request_sequence=1,
        enqueued_at=time.monotonic(),
    )

    ready_updates = [
        update
        for payload in node.publisher.payloads
        for update in payload.get("updates", [])
        if update.get("attribute_status") == "ready"
    ]
    assert ready_updates == []
    assert node.filter_counts["stale"] == 1
    assert node.episode_active is expected_active
    assert node.episode_generation == expected_generation
