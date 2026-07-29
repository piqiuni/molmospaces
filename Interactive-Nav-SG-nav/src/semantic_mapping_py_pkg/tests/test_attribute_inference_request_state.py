import json
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("rospy")

import interaction_attribute_inference_node as attribute_module
from interaction_attribute_inference_node import InteractionAttributeInferenceNode


class RecordingQueue:
    def __init__(self) -> None:
        self.discarded = []

    def discard(self, object_id, request_sequence=None) -> None:
        self.discarded.append((object_id, request_sequence))


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


def test_rle_only_minimal_gt_detection_passes_attribute_visibility_filter() -> None:
    node = object.__new__(InteractionAttributeInferenceNode)
    node.include_keywords = ("door", "fridge", "cabinet", "drawer")
    node.exclude_keywords = ("toilet", "sofa", "safe")
    node.min_visible_fraction = 0.20
    node.min_visible_pixels = 64
    node.min_bbox_area_px = 512
    node.max_distance_m = 6.0
    node.required_consecutive_observations = 2
    detection = {
        "id": "door_1",
        "name": "Door",
        "bbox_2d": [0, 0, 31, 31],
        # The V3 minimal-GT wire format contains only compact RLE, not the
        # old dense `segmentation` or `mask` aliases.
        "mask_rle": {"size": [32, 32], "counts": [0, 512, 512]},
        "box_3d": {"center": [1.0, 2.0, 1.0], "size": [0.2, 1.0, 2.0]},
    }

    assert node._passes_observation_filter(detection)
    assert InteractionAttributeInferenceNode._capture_step(
        {"capture_step": 17, "frame_index": 2}
    ) == 17


def test_uncertain_portal_result_retries_after_short_refresh_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = object.__new__(InteractionAttributeInferenceNode)
    node.lock = threading.Lock()
    node.pending = {}
    node.completed = {
        "door_1": {
            "signature": "door-evidence",
            "completed_at": 10.0,
            "refresh_interval_s": 5.0,
        }
    }
    node.last_request = {}
    node.min_interval_s = 0.0
    node.request_sequence = 0
    node.generations = {"door_1": 0}
    node.current_episode_id = "episode_1"
    node.success_refresh_interval_s = 120.0
    node.uncertain_portal_refresh_interval_s = 5.0
    node.uncertain_portal_confidence = 0.65

    monkeypatch.setattr(attribute_module.time, "monotonic", lambda: 15.1)
    reservation = node._try_reserve("door_1", "door-evidence")
    assert reservation is not None
    assert node._attribute_refresh_interval(
        {"name": "door"},
        {
            "interaction_class": "unknown",
            "coarse_state": "unknown",
            "confidence": 0.2,
        },
    ) == 5.0
