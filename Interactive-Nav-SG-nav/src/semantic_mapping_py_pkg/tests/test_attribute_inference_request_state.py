import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rospy")

import interaction_attribute_inference_node as attribute_module
from interaction_attribute_inference_node import InteractionAttributeInferenceNode


class RecordingQueue:
    def __init__(self, discarded_payloads=None) -> None:
        self.discarded = []
        self.discarded_payloads = list(discarded_payloads or [])

    def discard(self, object_id, request_sequence=None):
        self.discarded.append((object_id, request_sequence))
        return list(self.discarded_payloads)


class RecordingClient:
    def __init__(self) -> None:
        self.config = SimpleNamespace(model="test-mllm")
        self.calls = []

    def request_json(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            error="",
            payload={
                "object_id": "target",
                "interactable": True,
                "interaction_class": "container",
                "coarse_state": "closed",
                "portal_morphology": None,
                "portal_aperture_evidence": None,
                "view_state": "front",
                "view_state_confidence": 0.9,
                "front_surface_visible": True,
                "front_surface_confidence": 0.9,
                "approach_ready": True,
                "needs_reobserve": False,
                "interaction_parts": [],
                "confidence": 0.9,
            },
        )


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


def test_public_detector_border_flag_marks_only_clipped_boxes() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert InteractionAttributeInferenceNode._detection_bbox_touches_image_border(
        image, {"bbox_2d": [40, 20, 160, 80]}
    ) is False
    assert InteractionAttributeInferenceNode._detection_bbox_touches_image_border(
        image, {"bbox_2d": [0, 20, 80, 80]}
    ) is True
    assert InteractionAttributeInferenceNode._detection_bbox_touches_image_border(
        image, {"bbox_2d": [120, 20, 200, 80]}
    ) is True
    assert InteractionAttributeInferenceNode._detection_bbox_border_edges(
        image, {"bbox_2d": [40, 20, 160, 80]}
    ) == []
    assert InteractionAttributeInferenceNode._detection_bbox_border_edges(
        image, {"bbox_2d": [0, 20, 80, 80]}
    ) == ["left"]
    assert InteractionAttributeInferenceNode._detection_bbox_border_edges(
        image, {"bbox_2d": [40, 70, 160, 100]}
    ) == ["bottom"]


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


def test_targeted_refresh_requires_later_capture_and_rgb_sequence() -> None:
    node = object.__new__(InteractionAttributeInferenceNode)
    node.lock = threading.Lock()
    node.aliases = {"alias_1": "object_1", "object_1": "object_1"}
    node.targeted_refresh_requests = {
        "object_1": {
            "object_id": "alias_1",
            "episode_id": "episode_1",
            "minimum_capture_step": 12,
            "minimum_image_sequence": 7,
            "refresh_sequence": 3,
            "reason": "need_front_view",
        }
    }
    detection = {"id": "alias_1", "bbox_2d": [10, 10, 40, 40]}

    assert (
        node._targeted_refresh_for_detection(
            "object_1",
            detection,
            episode_id="episode_1",
            capture_step=12,
            image_sequence=8,
        )
        is None
    )
    assert (
        node._targeted_refresh_for_detection(
            "object_1",
            detection,
            episode_id="episode_1",
            capture_step=13,
            image_sequence=7,
        )
        is None
    )
    matched = node._targeted_refresh_for_detection(
        "object_1",
        detection,
        episode_id="episode_1",
        capture_step=13,
        image_sequence=8,
    )
    assert matched is not None
    assert matched["reason"] == "need_front_view"


def test_targeted_refresh_parses_only_public_container_or_portal_constraints() -> None:
    valid = InteractionAttributeInferenceNode._parse_targeted_refresh_payload(
        {
            "object_id": "fridge_1",
            "episode_id": "episode_1",
            "minimum_capture_step": 12,
            "reason": "mllm_container_pre_action_visual",
            "request_id": "decision_1:m1:001",
            "expected_node_type": "CONTAINER",
        }
    )

    assert valid is not None
    assert valid["expected_node_type"] == "container"
    assert (
        InteractionAttributeInferenceNode._parse_targeted_refresh_payload(
            {
                "object_id": "fridge_1",
                "minimum_capture_step": 12,
                "expected_node_type": "room",
            }
        )
        is None
    )


def test_targeted_refresh_replaces_stale_request_and_keeps_tracking_fields() -> None:
    node = object.__new__(InteractionAttributeInferenceNode)
    node.lock = threading.Lock()
    node.current_episode_id = "episode_1"
    node.pending = {"object_1": {"request_sequence": 4, "generation": 2}}
    node.generations = {"object_1": 2}
    node.completed = {"object_1": {"signature": "old"}}
    node.last_request = {"object_1": 1.0}
    node.request_sequence = 5
    node.request_queue = RecordingQueue()
    node.filter_counts = {"targeted_refresh_matched": 0}
    published = []
    node._publish_updates = lambda _episode, _stamp, updates: published.extend(updates)
    refresh = {
        "request_key": "object_1",
        "request_id": "refresh_9",
        "refresh_sequence": 9,
        "reason": "need_front_view",
        "minimum_capture_step": 20,
    }

    reservation = node._force_reserve_targeted_refresh(
        "object_1", "fresh", "episode_1", refresh
    )

    assert reservation == {"generation": 3, "request_sequence": 6}
    assert node.request_queue.discarded == [("object_1", 4)]
    assert node.pending["object_1"]["targeted_refresh"]["request_id"] == "refresh_9"
    assert published[0]["attribute_status"] == "stale"
    assert published[0]["request_sequence"] == 4
    status = node._attribute_status_patch(
        {
            "object_id": "object_1",
            "frame_id": "21",
            "image_sequence": 18,
            "request_sequence": 6,
            "signature": "fresh",
            "targeted_refresh": refresh,
        },
        "ready",
    )
    assert status["targeted_refresh"] is True
    assert status["targeted_refresh_request_id"] == "refresh_9"
    assert status["targeted_refresh_image_sequence"] == 18


def test_changed_state_marks_discarded_queued_request_stale() -> None:
    old_request = {
        "object_id": "object_1",
        "episode_id": "episode_1",
        "frame_id": "8",
        "image_sequence": 9,
        "stamp": 10.0,
        "signature": "old",
        "generation": 0,
        "request_sequence": 4,
        "enqueued_at": 1.0,
        "targeted_refresh": {},
    }
    node = object.__new__(InteractionAttributeInferenceNode)
    node.lock = threading.Lock()
    node.pending = {
        "object_1": {
            "request_sequence": 4,
            "signature": "old",
            "generation": 0,
            "episode_id": "episode_1",
        }
    }
    node.generations = {"object_1": 0}
    node.last_request = {"object_1": 1.0}
    node.filter_counts = {"coalesced": 0}
    node.request_queue = RecordingQueue([old_request])
    published = []
    node._publish_updates = lambda _episode, _stamp, updates: published.extend(updates)

    node._invalidate_if_state_changed("object_1", "new", "episode_1")

    assert node.request_queue.discarded == [("object_1", 4)]
    assert node.filter_counts["coalesced"] == 1
    assert published == [
        {
            "object_id": "object_1",
            "attribute_status": "stale",
            "observation_capture_step": 8,
            "request_sequence": 4,
            "observation_signature": "old",
            "source": "mllm_attribute_inference",
            "error": "queue_replaced_by_newer_object_evidence",
        }
    ]


def test_attribute_visual_evidence_is_full_image_with_target_outline_and_inset() -> None:
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    image[25:85, 60:110] = (30, 100, 200)
    evidence = InteractionAttributeInferenceNode._compose_attribute_visual_evidence(
        image,
        {"bbox_2d": [60, 25, 110, 85]},
        margin_ratio=0.10,
    )

    assert evidence is not None
    assert evidence.shape == image.shape
    assert not np.array_equal(evidence, image)
    # The BGR yellow outline gives an unambiguous target anchor to M1.
    assert np.any(np.all(evidence == np.array([0, 255, 255]), axis=2))


def test_portal_visual_evidence_keeps_full_frame_without_crop_inset() -> None:
    image = np.full((100, 160, 3), 17, dtype=np.uint8)
    image[25:85, 60:110] = (30, 100, 200)
    detection = {
        "semantic_name": "door",
        "bbox_2d": [60, 25, 110, 85],
    }

    evidence = InteractionAttributeInferenceNode._compose_attribute_visual_evidence(
        image,
        detection,
        margin_ratio=0.10,
        include_crop_inset=False,
    )

    assert InteractionAttributeInferenceNode._is_portal_detection(detection) is True
    assert evidence is not None
    assert evidence.shape == image.shape
    # Every corner stays from the original full frame; only the target outline
    # is added, so wall/leaf/aperture context is never replaced by an inset.
    for row, col in ((0, 0), (0, 159), (99, 0), (99, 159)):
        assert np.array_equal(evidence[row, col], image[row, col])
    assert np.any(np.all(evidence == np.array([0, 255, 255]), axis=2))


def test_target_bbox_containment_requires_visible_target_mask_inside_box() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    contained = InteractionAttributeInferenceNode._target_bbox_containment(
        image,
        {
            "bbox_2d": [8, 4, 20, 16],
            # Geometry/restricted-GT boxes use inclusive maxima; pixels on the
            # right/bottom edges are therefore still contained.
            "segmentation": {"rows": [5, 8, 16], "cols": [9, 12, 20]},
        },
    )
    outside = InteractionAttributeInferenceNode._target_bbox_containment(
        image,
        {
            "bbox_2d": [8, 4, 20, 16],
            "segmentation": {"rows": [5, 8], "cols": [9, 22]},
        },
    )

    assert contained["valid"] is True
    assert contained["target_mask_pixels"] == 3
    assert outside["valid"] is False
    assert outside["reason"] == "target_mask_outside_bbox"


def test_invalid_target_bbox_releases_armed_refresh_with_terminal_failure() -> None:
    node = object.__new__(InteractionAttributeInferenceNode)
    node.lock = threading.Lock()
    node.filter_counts = {"targeted_refresh_bbox_rejected": 0, "failed": 0}
    refresh = {
        "request_key": "fridge_1",
        "refresh_sequence": 3,
        "request_id": "decision:m1:003",
        "minimum_capture_step": 40,
        "reason": "container_front",
    }
    node.targeted_refresh_requests = {"fridge_1": dict(refresh)}
    published = []
    node._publish_updates = lambda episode, stamp, updates: published.append(
        (episode, stamp, updates)
    )

    node._reject_targeted_refresh_bbox(
        object_id="fridge_1",
        episode_id="episode_1",
        observation_stamp=12.0,
        frame_id="44",
        image_sequence=45,
        signature="view-44",
        targeted_refresh=refresh,
        bbox_containment={"valid": False, "reason": "target_mask_unavailable"},
    )

    assert node.targeted_refresh_requests == {}
    patch = published[0][2][0]
    assert patch["attribute_status"] == "failed"
    assert patch["targeted_refresh_request_id"] == "decision:m1:003"
    assert patch["observation_capture_step"] == 44
    assert patch["error"].endswith("target_mask_unavailable")


def test_target_multiview_history_requires_step_and_pose_separation() -> None:
    evidence = np.zeros((8, 8, 3), dtype=np.uint8)
    history = [
        {
            "capture_step": 10,
            "frame_id": "10",
            "observation_pose_xyyaw": [0.0, 0.0, 0.0],
            "bbox_containment": {"valid": True},
            "visual_evidence": evidence,
        },
        {
            # Enough steps, but effectively the same viewpoint: reject.
            "capture_step": 20,
            "frame_id": "20",
            "observation_pose_xyyaw": [0.04, 0.01, 0.05],
            "bbox_containment": {"valid": True},
            "visual_evidence": evidence,
        },
        {
            "capture_step": 34,
            "frame_id": "34",
            "observation_pose_xyyaw": [0.35, 0.0, 0.0],
            "bbox_containment": {"valid": True},
            "visual_evidence": evidence,
        },
        {
            "capture_step": 50,
            "frame_id": "50",
            "observation_pose_xyyaw": [0.35, 0.30, 0.4],
            "bbox_containment": {"valid": True},
            "visual_evidence": evidence,
        },
    ]

    selected = InteractionAttributeInferenceNode._select_diverse_target_visual_history(
        history,
        max_images=3,
        min_step_gap=8,
        min_position_gap_m=0.25,
        min_yaw_gap_rad=0.25,
    )

    assert [item["capture_step"] for item in selected] == [10, 34, 50]


def test_m1_inference_sends_only_opaque_context_and_one_composite_image() -> None:
    node = object.__new__(InteractionAttributeInferenceNode)
    node.lock = threading.Lock()
    node.current_episode_id = "episode_1"
    node.pending = {
        "object_1": {
            "request_sequence": 1,
            "generation": 0,
            "episode_id": "episode_1",
        }
    }
    node.generations = {"object_1": 0}
    node.last_request = {}
    node.completed = {}
    node.filter_counts = {"started": 0, "stale": 0, "completed": 0, "failed": 0}
    node.visual_evidence_max_side_px = 0
    node.request_timeout_s = 1.0
    node.max_output_tokens = 256
    node.success_refresh_interval_s = 120.0
    node.client = RecordingClient()
    published = []
    node._publish_updates = lambda episode_id, stamp, updates: published.append(updates)
    node._publish_status = lambda: None
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    visual_evidence = node._compose_attribute_visual_evidence(
        image,
        {"bbox_2d": [30, 15, 90, 70]},
        margin_ratio=0.08,
    )
    assert visual_evidence is not None

    node._infer(
        object_id="object_1",
        detection={
            "name": "fridge",
            "category": "private_gt_category",
            "position": [9.0, 8.0, 7.0],
        },
        visual_evidence=visual_evidence,
        episode_id="episode_1",
        frame_id="14",
        image_sequence=22,
        stamp=10.0,
        signature="fresh",
        generation=0,
        request_sequence=1,
        enqueued_at=0.0,
        targeted_refresh={},
    )

    request = node.client.calls[0]
    assert request["context"] == {"object_id": "target"}
    assert len(request["images"]) == 1
    assert request["response_schema"]["schema"]["properties"]["object_id"]["enum"] == [
        "target"
    ]
    assert published[0][0]["object_id"] == "object_1"
    assert published[0][0]["approach_ready"] is True


def test_targeted_container_refresh_constrains_class_without_supplying_view() -> None:
    class PortalFallbackClient(RecordingClient):
        def request_json(self, **kwargs):
            self.calls.append(kwargs)
            # Deliberately violate the targeted strict schema to exercise the
            # compatibility guard used with older OpenAI-compatible servers.
            return SimpleNamespace(
                error="",
                payload={
                    "object_id": "target",
                    "interactable": True,
                    "interaction_class": "portal",
                    "coarse_state": "static_open",
                    "portal_morphology": {"door_leaf": "absent", "confidence": 0.9},
                    "portal_aperture_evidence": {
                        "open_aperture": "visible",
                        "confidence": 0.9,
                    },
                    "view_state": "front",
                    "view_state_confidence": 0.9,
                    "front_surface_visible": True,
                    "front_surface_confidence": 0.9,
                    "approach_ready": True,
                    "needs_reobserve": False,
                    "interaction_parts": [],
                    "confidence": 0.9,
                },
            )

    node = object.__new__(InteractionAttributeInferenceNode)
    node.lock = threading.Lock()
    node.current_episode_id = "episode_1"
    node.pending = {
        "fridge_1": {
            "request_sequence": 1,
            "generation": 0,
            "episode_id": "episode_1",
        }
    }
    node.generations = {"fridge_1": 0}
    node.last_request = {}
    node.completed = {}
    node.filter_counts = {"started": 0, "stale": 0, "completed": 0, "failed": 0}
    node.visual_evidence_max_side_px = 0
    node.request_timeout_s = 1.0
    node.max_output_tokens = 256
    node.success_refresh_interval_s = 120.0
    node.client = PortalFallbackClient()
    published = []
    node._publish_updates = lambda _episode, _stamp, updates: published.extend(updates)
    node._publish_status = lambda: None

    node._infer(
        object_id="fridge_1",
        detection={"name": "fridge"},
        visual_evidence=np.zeros((40, 60, 3), dtype=np.uint8),
        episode_id="episode_1",
        frame_id="14",
        image_sequence=22,
        stamp=10.0,
        signature="fresh",
        generation=0,
        request_sequence=1,
        enqueued_at=0.0,
        targeted_refresh={
            "request_id": "decision_1:m1:001",
            "refresh_sequence": 1,
            "reason": "mllm_container_pre_action_visual",
            "minimum_capture_step": 12,
            "expected_node_type": "container",
        },
    )

    request = node.client.calls[0]
    assert request["context"] == {
        "object_id": "target",
        "expected_node_type": "container",
    }
    instruction = request["instruction"]
    assert "Judge frontality primarily from the target's horizontal perspective" in instruction
    assert "lower boundary may leave the frame" in instruction
    assert "vertical framing/height cues alone are not evidence of an oblique view" in instruction
    assert "do not require the top or bottom boundary to be fully visible" in instruction
    assert "include only action regions that are actually visible" in instruction
    properties = request["response_schema"]["schema"]["properties"]
    assert properties["interaction_class"]["enum"] == ["container"]
    assert properties["portal_morphology"] == {"type": "null"}
    patch = published[0]
    assert patch["interaction_class"] == "container"
    assert patch["coarse_state"] == "unknown"
    assert patch["m1_reported_interaction_class"] == "portal"
    assert patch["view_state"] == "front"
    assert "portal_morphology" not in patch
    assert "portal_aperture_evidence" not in patch


def test_m1_request_expired_in_local_queue_is_not_sent() -> None:
    node = object.__new__(InteractionAttributeInferenceNode)
    node.lock = threading.Lock()
    node.current_episode_id = "episode_1"
    node.pending = {
        "object_1": {
            "request_sequence": 1,
            "generation": 0,
            "episode_id": "episode_1",
        }
    }
    node.generations = {"object_1": 0}
    node.last_request = {}
    node.completed = {}
    node.filter_counts = {
        "started": 0,
        "stale": 0,
        "completed": 0,
        "expired": 0,
        "failed": 0,
    }
    node.visual_evidence_max_side_px = 0
    node.request_timeout_s = 1.0
    node.max_output_tokens = 256
    node.success_refresh_interval_s = 120.0
    node.client = RecordingClient()
    published = []
    node._publish_updates = lambda _episode, _stamp, updates: published.extend(updates)
    node._publish_status = lambda: None

    node._infer(
        object_id="object_1",
        detection={"name": "fridge"},
        visual_evidence=np.zeros((40, 60, 3), dtype=np.uint8),
        episode_id="episode_1",
        frame_id="4",
        image_sequence=5,
        stamp=10.0,
        signature="fresh",
        generation=0,
        request_sequence=1,
        enqueued_at=attribute_module.time.monotonic() - 2.0,
        targeted_refresh={},
        deadline_monotonic=attribute_module.time.monotonic() - 1.0,
    )

    assert node.client.calls == []
    assert node.filter_counts["expired"] == 1
    assert node.filter_counts["failed"] == 0
    assert node.last_request == {}
    assert published[-1]["attribute_status"] == "failed"
    assert published[-1]["error"] == "queue_deadline_expired_before_send"
