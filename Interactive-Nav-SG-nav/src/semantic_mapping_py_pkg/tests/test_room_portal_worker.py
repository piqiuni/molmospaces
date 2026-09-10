"""Actual mapper handoff under a busy room lock, without a ROS master."""
from collections import deque
from types import SimpleNamespace
import threading

import pytest

pytest.importorskip("rospy")
from semantic_mapping_node import SemanticMappingNode
from semantic_mapping_py_pkg.room_segmentation import RoomSegmenter


def make_node():
    node = object.__new__(SemanticMappingNode)
    node.lock = threading.RLock()
    node._room_lock = threading.RLock()
    node._room_work_condition = threading.Condition()
    node._room_worker_thread = object()
    node._room_worker_stopping = False
    node._room_work_pending = None
    node._room_portal_pending = deque(maxlen=64)
    node._room_portal_dropped_batches = 0
    node._room_epoch = node._room_state_epoch = 0
    node._room_input_revision = node._room_topology_revision = 0
    node.room_segmenter = RoomSegmenter(room_portal_detector_min_confirmations=2)
    node._room_segmentation_overlay = SimpleNamespace(reset=lambda: None)
    node.latest_occupancy_grid = None
    return node


def door(name="door1", x=1.0):
    return {"id": name, "name": "door", "position": [x, 2, 1],
            "aabb_size": [1, 0.1, 2]}


def test_detection_and_reset_do_not_wait_for_segmentation_lock():
    node = make_node()
    completed = threading.Event()
    errors = []

    def callback():
        try:
            for _ in range(2):
                node._update_room_portal_hints([door()], source_mode="detector_online", epoch=0)
            node._reset_room_worker_state()
        except Exception as exc:
            errors.append(exc)
        finally:
            completed.set()

    with node._room_lock:
        thread = threading.Thread(target=callback, daemon=True)
        thread.start()
        finished_while_busy = completed.wait(timeout=1.0)
    thread.join(timeout=1.0)
    assert finished_while_busy, "portal/reset callback waited for room segmentation"
    assert not errors
    assert not node.room_segmenter.state.portal_hints
    node._consume_room_portal_hints(0)
    hint = node.room_segmenter.state.portal_hints["door1"]
    assert hint["active"] and hint["confirmations"] == 2
    assert node._room_topology_revision == 1


def test_reset_discards_old_hints_and_pending_open_requests():
    node = make_node()
    node._update_room_portal_hints([door("old")], source_mode="detector_online", epoch=0)
    node._enqueue_room_refresh(post_open_result={"node_id": "old"}, force_stable=True)
    node.room_segmenter.update_portal_hints([door("active-old")], "realtime_gt_observation")
    with node.lock:
        node._room_epoch = 1
    node._reset_room_worker_state()
    assert node._room_work_pending["epoch"] == 1
    assert node._room_work_pending["post_open_results"] == []
    assert not node._room_work_pending["force_stable"]
    node._update_room_portal_hints([door("late-old")], source_mode="detector_online", epoch=0)
    node._update_room_portal_hints([door("new")], source_mode="realtime_gt_observation", epoch=1)
    node._consume_room_portal_hints(0)  # A stale job cannot drain new evidence.
    assert len(node._room_portal_pending) == 2
    # Even without an OCC yet, a new-epoch job must reset and consume hints.
    node._process_room_refresh_request(node._room_work_pending)
    assert set(node.room_segmenter.state.portal_hints) == {"new"}
    assert node._room_state_epoch == 1


def test_handoff_is_bounded_and_preserves_recent_observation_order():
    node = make_node()
    for index in range(100):
        node._update_room_portal_hints([door(str(index))], source_mode="detector_online", epoch=0)
    assert len(node._room_portal_pending) == 64
    assert node._room_portal_dropped_batches == 36
    node._consume_room_portal_hints(0)
    assert list(node.room_segmenter.state.portal_hints) == [str(i) for i in range(36, 100)]


def test_non_portal_observations_do_not_schedule_segmentation():
    node = make_node()
    node._update_room_portal_hints([{"name": "bottle"}], source_mode="detector_online", epoch=0)
    assert node._room_work_pending is None
    assert not node._room_portal_pending


def test_same_epoch_requests_preserve_open_evidence():
    node = make_node()
    node._enqueue_room_refresh(post_open_result={"node_id": "door1"}, force_stable=True)
    node._enqueue_room_refresh(reason="occupancy")
    assert node._room_work_pending["post_open_results"] == [{"node_id": "door1"}]
    assert node._room_work_pending["force_stable"]


def test_old_occupancy_callback_cannot_relabel_its_open_result_after_reset():
    node = make_node()
    node._room_epoch = 1
    node._enqueue_room_refresh(reason="episode_reset", epoch=1)
    pending = node._room_work_pending
    assert not node._enqueue_room_refresh(
        post_open_result={"node_id": "old-door"}, force_stable=True, epoch=0,
    )
    assert node._room_work_pending is pending
    assert pending["post_open_results"] == []
