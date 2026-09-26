"""Exercise real room inference methods without ROS services or a model server."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
from time import monotonic as real_monotonic

import pytest

pytest.importorskip("rospy")
import interaction_attribute_inference_node as module
from interaction_attribute_inference_node import InteractionAttributeInferenceNode
from semantic_mapping_py_pkg.attribute_inference_queue import LatestPriorityRequestQueue


class ClockEvent:
    def __init__(self, lock):
        self.now = 100.0
        self.stopped = False
        self.on_wait = None
        self.lock = lock

    def is_set(self):
        return self.stopped

    def wait(self, duration):
        assert self.lock.acquire(blocking=False), "callback lock held during throttling"
        self.lock.release()
        self.now += duration
        if self.on_wait:
            self.on_wait()
        return self.stopped


def test_room_input_excludes_structural_labels_without_losing_furniture():
    labels = ["door", "wooden door", "bay_window", "office-window", "combination_lock",
              "chair", "office_desk", "fridge", "window_air_conditioner"]
    objects = [{"object_id": str(i), "name": "track_" + str(i), "category": label}
               for i, label in enumerate(labels)]
    filtered = InteractionAttributeInferenceNode._room_objects(
        objects, excluded_labels=["door", "window", "lock"])
    assert {x["category"] for x in filtered} == set(labels[5:])
    assert len(InteractionAttributeInferenceNode._room_objects(objects)) == len(objects)

@pytest.fixture
def runtime(monkeypatch):
    node = object.__new__(InteractionAttributeInferenceNode)
    node.lock = threading.Lock()
    clock = ClockEvent(node.lock)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    node.shutdown_event = clock
    node.current_episode_id = "e1"
    node.room_dispatch_interval_s = 1.0
    node.room_next_dispatch_at = 0.0
    node.room_min_interval_s = 5.0
    node.room_success_refresh_interval_s = 120.0
    node.room_request_timeout_s = 8.0
    node.room_max_output_tokens = 96
    node.room_pending = {}
    node.room_completed = {}
    node.room_failures = {}
    node.room_failure_refresh_interval_s = 30.0
    node.room_fallback_enabled = False
    node.room_generations = {}
    node.room_last_request = {}
    node.room_request_sequence = 0
    node.room_counts = defaultdict(int)
    node.room_request_queue = LatestPriorityRequestQueue(2)
    node.updates = []
    node.calls = []
    node.on_model = None
    node._publish_room_updates = lambda episode, stamp, patches: node.updates.extend(patches)
    node._publish_status = lambda: None

    def model(**kwargs):
        node.calls.append((clock.now, kwargs))
        if node.on_model:
            node.on_model()
        return SimpleNamespace(error="", payload={
            "room_id": kwargs["context"]["room_id"],
            "room_attribute": "kitchen", "confidence": 0.9,
            "evidence_object_ids": ["stove"],
        })

    node.client = SimpleNamespace(config=SimpleNamespace(model="offline"), request_json=model)
    return node, clock


def reserve(node, clock, room_id=1, signature="stove"):
    room_key = f"room_{room_id}"
    reservation = node._try_reserve_room(room_key, signature)
    assert reservation is not None
    return dict(
        room_key=room_key, room_id=room_id, room_node_id=room_key,
        objects=[{"object_id": "stove", "name": signature}], episode_id="e1",
        room_box={"center_xy": [0.0, 0.0], "size_xy": [3.0, 4.0]},
        capture_step=12, stamp=123.0, signature=signature,
        enqueued_at=clock.now, deadline_monotonic=clock.now + 8.0,
        **reservation,
    )


def test_different_rooms_share_dispatch_budget_and_remain_text_only(runtime):
    node, clock = runtime
    first, second = reserve(node, clock, 1), reserve(node, clock, 2)
    node._infer_room(**first)
    node._infer_room(**second)
    assert [t for t, _ in node.calls] == pytest.approx([100.0, 101.0])
    assert node.room_counts["completed"] == 2
    assert node.updates[-1]["dispatch_wait_sec"] == pytest.approx(1.0)
    assert node.calls[-1][1]["timeout_s"] == pytest.approx(7.0)
    for _, call in node.calls:
        assert call["role"] == "room_attribute_inference"
        assert set(call["context"]) == {"room_id", "room_box", "capture_step", "objects", "episode_id"}
        assert not any("image" in key for key in call)


def test_failed_room_model_publishes_explicit_fallback_not_m1_ready(runtime):
    from semantic_mapping_py_pkg.room_inference_backends import WeightedRoomAttributeInferencer

    node, clock = runtime
    node.room_fallback_enabled = True
    node.room_fallback_inferencer = WeightedRoomAttributeInferencer(
        {"kitchen": {"stove": 1.0}}
    )
    node.client.request_json = lambda **kwargs: SimpleNamespace(
        error="ReadError", payload=None
    )
    node._infer_room(**reserve(node, clock))

    assert node.room_counts["failed"] == 1
    assert node.room_counts["completed"] == 0
    assert node.updates[-1]["room_attribute_status"] == "fallback"
    assert node.updates[-1]["fallback"] is True
    assert node.updates[-1]["error"] == "ReadError"
    assert node.updates[-1]["source"] == "weighted_object_types_fallback"
    assert "room_1" not in node.room_completed


def test_replaced_evidence_while_waiting_never_calls_model(runtime):
    node, clock = runtime
    payload = reserve(node, clock)
    node.room_next_dispatch_at = clock.now + 1.0
    clock.on_wait = lambda: node._invalidate_room_if_state_changed("room_1", "new", "e1")
    node._infer_room(**payload)
    assert not node.calls
    assert node.room_counts["stale"] == 1
    assert "room_1" not in node.room_last_request


def test_expiration_during_rate_wait_does_not_spend_slot(runtime):
    node, clock = runtime
    payload = reserve(node, clock)
    payload["deadline_monotonic"] = clock.now + 0.25
    node.room_next_dispatch_at = clock.now + 1.0
    node._infer_room(**payload)
    assert not node.calls
    assert node.room_next_dispatch_at == 101.0
    assert node.room_counts["expired"] == 1
    assert node.updates[-1]["error"] == "queue_deadline_expired_before_send"
    assert not node.room_pending


def test_shutdown_interrupts_rate_wait(runtime):
    node, clock = runtime
    payload = reserve(node, clock)
    node.room_next_dispatch_at = clock.now + 100.0
    clock.on_wait = lambda: setattr(clock, "stopped", True)
    node._infer_room(**payload)
    assert clock.now == pytest.approx(100.1)
    assert not node.calls


def test_new_episode_during_wait_discards_old_room_request(runtime):
    node, clock = runtime
    payload = reserve(node, clock)
    node.room_next_dispatch_at = clock.now + 1.0
    clock.on_wait = lambda: setattr(node, "current_episode_id", "e2")
    node._infer_room(**payload)
    assert not node.calls
    assert node.room_counts["stale"] == 1


def test_inflight_invalidation_keeps_retry_cooldown_and_rejects_old_result(runtime):
    node, clock = runtime
    payload = reserve(node, clock)
    node.on_model = lambda: node._invalidate_room_if_state_changed("room_1", "fridge", "e1")
    node._infer_room(**payload)
    assert len(node.calls) == 1
    assert not node.room_completed
    assert not any(patch.get("room_attribute_status") == "ready" for patch in node.updates)
    assert node.room_last_request["room_1"] == 100.0
    assert node._try_reserve_room("room_1", "fridge") is None
    clock.now += 5.0
    assert node._try_reserve_room("room_1", "fridge") is not None


def test_unchanged_success_stays_cached_after_dispatch_interval(runtime):
    node, clock = runtime
    node._infer_room(**reserve(node, clock))
    clock.now += 10.0
    assert node._try_reserve_room("room_1", "stove") is None
    assert len(node.calls) == 1


def test_disabled_dispatch_budget_preserves_unthrottled_other_profiles(runtime):
    node, clock = runtime
    node.room_dispatch_interval_s = 0.0
    node._infer_room(**reserve(node, clock, 1))
    node._infer_room(**reserve(node, clock, 2))
    assert [t for t, _ in node.calls] == [100.0, 100.0]


def test_concurrent_room_workers_share_the_same_gate(runtime, monkeypatch):
    node, clock = runtime
    monkeypatch.setattr(module.time, "monotonic", real_monotonic)
    node.shutdown_event = threading.Event()
    node.room_dispatch_interval_s = 0.05
    clock.now = real_monotonic()
    payloads = [reserve(node, clock, room_id) for room_id in (1, 2, 3)]
    starts = []
    model = node.client.request_json

    def timed_model(**kwargs):
        starts.append(real_monotonic())
        return model(**kwargs)

    node.client.request_json = timed_model
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(node._infer_room, **payload) for payload in payloads]
        for future in futures:
            future.result(timeout=3.0)
    assert len(starts) == 3
    assert all(second - first >= 0.04 for first, second in zip(starts, starts[1:]))
    assert node.room_counts["completed"] == 3
