"""Focused regressions for the optional web camera-overlay path."""

from __future__ import annotations

import errno
from http.server import ThreadingHTTPServer
import pathlib
import socket
import ssl
import sys
import threading
from unittest.mock import patch

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import physical_six_panel_server as server_module  # noqa: E402
from physical_six_panel_server import (  # noqa: E402
    _LatestRecordingWorker,
    PhysicalGateway,
    SixPanelRenderer,
    _ReusableHTTPServer,
)
from runtime_state import RuntimeState  # noqa: E402


def _frame(value: int) -> np.ndarray:
    return np.full((8, 12, 3), value, dtype=np.uint8)


def _set_current_frame(state: RuntimeState, sequence: int) -> None:
    with state._lock:
        state.frame_seq = sequence


def test_camera_overlay_selects_detection_capture_not_latest_rgb() -> None:
    state = RuntimeState()
    renderer = SixPanelRenderer(state)
    older, latest = _frame(10), _frame(11)
    renderer.remember_camera_frame(
        10, older, transport_session="sensor-a", transport_connection=1
    )
    renderer.remember_camera_frame(
        11, latest, transport_session="sensor-a", transport_connection=1
    )
    _set_current_frame(state, 11)
    renderer.remember_camera_detections({
        "seq": 10,
        "detections": [{
            "capture_seq": 10,
            "semantic_class": "door",
            "bbox": [1, 1, 6, 6],
        }],
    })

    key, paired_rgb, detections = renderer.camera_overlay_candidate()

    assert key[:3] == ("exact", ("sensor-a", 1), 10)
    assert paired_rgb is older
    assert paired_rgb is not latest
    assert detections[0]["capture_seq"] == 10


def test_late_same_frame_detections_change_overlay_render_key() -> None:
    state = RuntimeState()
    renderer = SixPanelRenderer(state)
    rgb = _frame(21)
    renderer.remember_camera_frame(
        21, rgb, transport_session="sensor-a", transport_connection=1
    )
    _set_current_frame(state, 21)

    raw_key, raw_rgb, raw_detections = renderer.camera_overlay_candidate()
    assert raw_key[0] == "raw"
    assert raw_rgb is rgb
    assert raw_detections == []

    renderer.remember_camera_detections({
        "seq": 21,
        "detections": [{"capture_seq": 21, "bbox": [1, 1, 3, 3]}],
    })
    first_key, _, first_detections = renderer.camera_overlay_candidate()
    assert first_key[0] == "exact"
    assert first_key != raw_key
    assert first_detections[0]["bbox"] == [1, 1, 3, 3]

    # A corrected/repeated report for the same capture is still a new render
    # input.  This is the case the old RGB-only loop silently skipped.
    renderer.remember_camera_detections({
        "seq": 21,
        "detections": [{"capture_seq": 21, "bbox": [2, 2, 7, 7]}],
    })
    corrected_key, _, corrected_detections = renderer.camera_overlay_candidate()
    assert corrected_key[0] == "exact"
    assert corrected_key != first_key
    assert corrected_detections[0]["bbox"] == [2, 2, 7, 7]


def test_detection_can_arrive_before_first_web_rgb_decode() -> None:
    state = RuntimeState()
    renderer = SixPanelRenderer(state)
    renderer.remember_camera_detections({
        "seq": 5,
        "detections": [{"capture_seq": 5, "bbox": [1, 1, 3, 3]}],
    })
    rgb = _frame(5)
    renderer.remember_camera_frame(
        5, rgb, transport_session="sensor-a", transport_connection=1
    )
    _set_current_frame(state, 5)

    key, paired_rgb, detections = renderer.camera_overlay_candidate()

    assert key[:3] == ("exact", ("sensor-a", 1), 5)
    assert paired_rgb is rgb
    assert detections[0]["capture_seq"] == 5


def test_camera_history_is_bounded_and_never_borrows_mismatched_boxes() -> None:
    state = RuntimeState()
    renderer = SixPanelRenderer(state)
    renderer._camera_frame_capacity = 3
    for sequence in range(1, 5):
        renderer.remember_camera_frame(
            sequence,
            _frame(sequence),
            transport_session="sensor-a",
            transport_connection=1,
        )
    _set_current_frame(state, 4)
    renderer.remember_camera_detections({
        "seq": 1,
        "detections": [{"capture_seq": 1, "bbox": [0, 0, 2, 2]}],
    })

    key, paired_rgb, detections = renderer.camera_overlay_candidate()

    assert len(renderer._camera_frames) == 3
    assert 1 not in renderer._camera_frames
    assert key[0] == "raw"
    assert int(paired_rgb[0, 0, 0]) == 4
    assert detections == []


def test_transport_generation_does_not_reuse_retired_detection_sequence() -> None:
    state = RuntimeState()
    renderer = SixPanelRenderer(state)
    renderer.remember_camera_frame(
        1, _frame(10), transport_session="old", transport_connection=1
    )
    _set_current_frame(state, 1)
    renderer.remember_camera_detections({
        "seq": 1,
        "detections": [{"capture_seq": 1, "bbox": [0, 0, 2, 2]}],
    })
    old_key = renderer.camera_overlay_candidate()[0]
    assert old_key[0] == "exact"
    assert renderer.camera_overlay_key_is_current(old_key)

    new_rgb = _frame(99)
    renderer.remember_camera_frame(
        1, new_rgb, transport_session="new", transport_connection=1
    )
    key, paired_rgb, detections = renderer.camera_overlay_candidate()
    assert key[0] == "raw"
    assert paired_rgb is new_rgb
    assert detections == []
    assert not renderer.camera_overlay_key_is_current(old_key)

    renderer.remember_camera_detections({
        "seq": 1,
        "detections": [{"capture_seq": 1, "bbox": [1, 1, 3, 3]}],
    })
    key, paired_rgb, detections = renderer.camera_overlay_candidate()
    assert key[:3] == ("exact", ("new", 1), 1)
    assert paired_rgb is new_rgb
    assert detections[0]["bbox"] == [1, 1, 3, 3]


def test_explicit_detection_generation_handles_detection_before_new_rgb() -> None:
    state = RuntimeState()
    renderer = SixPanelRenderer(state)
    old_rgb, new_rgb = _frame(30), _frame(31)
    renderer.remember_camera_frame(
        1, old_rgb, transport_session="old", transport_connection=1
    )
    _set_current_frame(state, 1)
    renderer.remember_camera_detections({
        "seq": 1,
        "transport_session": "old",
        "transport_connection": 1,
        "detections": [{"capture_seq": 1, "bbox": [0, 0, 2, 2]}],
    })
    assert renderer.camera_overlay_candidate()[0][0] == "exact"

    # The new detector receipt can arrive before the web mirror has decoded
    # the corresponding RGB.  It must neither borrow old RGB nor be made stale
    # merely because the frame-generation transition happens afterwards.
    renderer.remember_camera_detections({
        "seq": 1,
        "transport_session": "new",
        "transport_connection": 2,
        "detections": [{"capture_seq": 1, "bbox": [1, 1, 4, 4]}],
    })
    key, paired_rgb, detections = renderer.camera_overlay_candidate()
    assert key[0] == "raw"
    assert paired_rgb is old_rgb
    assert detections == []

    renderer.remember_camera_frame(
        1, new_rgb, transport_session="new", transport_connection=2
    )
    key, paired_rgb, detections = renderer.camera_overlay_candidate()
    assert key[:3] == ("exact", ("new", 2), 1)
    assert paired_rgb is new_rgb
    assert detections[0]["bbox"] == [1, 1, 4, 4]


def test_delayed_retired_detection_never_matches_reused_new_sequence() -> None:
    state = RuntimeState()
    renderer = SixPanelRenderer(state)
    new_rgb = _frame(41)
    renderer.remember_camera_frame(
        1, new_rgb, transport_session="new", transport_connection=2
    )
    _set_current_frame(state, 1)

    renderer.remember_camera_detections({
        "seq": 1,
        "transport_session": "old",
        "transport_connection": 1,
        "detections": [{"capture_seq": 1, "bbox": [0, 0, 2, 2]}],
    })
    key, paired_rgb, detections = renderer.camera_overlay_candidate()

    assert key[0] == "raw"
    assert paired_rgb is new_rgb
    assert detections == []


def test_composite_rejects_same_sequence_from_retired_generation() -> None:
    state = RuntimeState()
    renderer = SixPanelRenderer(state)
    renderer.remember_camera_frame(
        1, _frame(50), transport_session="new", transport_connection=2
    )
    detections = [{"capture_seq": 1, "bbox": [0, 0, 2, 2]}]

    assert not renderer.camera_detections_match_current_frame(
        1,
        {
            "seq": 1,
            "transport_session": "old",
            "transport_connection": 1,
        },
        detections,
    )
    assert renderer.camera_detections_match_current_frame(
        1,
        {
            "seq": 1,
            "transport_session": "new",
            "transport_connection": 2,
        },
        detections,
    )


def test_web_sensor_mirror_decodes_only_rgb_and_keeps_depth_png() -> None:
    gateway = object.__new__(PhysicalGateway)
    gateway.state = RuntimeState()
    gateway.renderer = SixPanelRenderer(gateway.state)
    gateway._sensor_lock = threading.Lock()
    gateway._active_sensor_session = "sensor-a"
    gateway._active_sensor_connection = 1
    rgb = _frame(7)
    decoded_encodings: list[str] = []

    def fake_decode(_value: str, encoding: str) -> np.ndarray:
        decoded_encodings.append(encoding)
        assert encoding == "jpeg"
        return rgb

    packet = {
        "seq": 7,
        "_bridge_seq": 17,
        "_transport_session": "sensor-a",
        "_transport_connection": 1,
        "stamp": 12.5,
        "rgb": {"data": "rgb-jpeg"},
        "depth": {"data": "depth-png16"},
        "depth_scale": 0.001,
        "intrinsics": {"width": 12, "height": 8},
    }
    with patch.object(server_module, "_decode", side_effect=fake_decode):
        gateway._process_sensor_frame(packet)

    assert decoded_encodings == ["jpeg"]
    assert gateway.state.rgb is rgb
    assert gateway.state.depth is None
    raw = gateway.state.raw_frame()
    assert raw["rgb"] == "rgb-jpeg"
    assert raw["depth"] == "depth-png16"
    assert raw["seq"] == 17


def test_sensor_generation_transition_cannot_split_check_and_frame_commit() -> None:
    gateway = object.__new__(PhysicalGateway)
    gateway.state = RuntimeState()
    gateway.renderer = SixPanelRenderer(gateway.state)
    gateway._sensor_lock = threading.Lock()
    gateway._active_sensor_session = "old"
    gateway._active_sensor_connection = 1
    gateway._retired_sensor_sessions = set()
    gateway._sensor_source_seq = -1
    gateway._latest_sensor_bridge_seq = -1
    gateway._latest_sensor_packet = None
    entered_commit = threading.Event()
    release_commit = threading.Event()
    original_update = gateway.state.update_frame

    def blocking_update(**kwargs):
        assert gateway._sensor_lock.locked()
        entered_commit.set()
        assert release_commit.wait(2.0)
        original_update(**kwargs)

    gateway.state.update_frame = blocking_update
    old_packet = {
        "seq": 1,
        "_bridge_seq": 1,
        "_transport_session": "old",
        "_transport_connection": 1,
        "stamp": 10.0,
        "rgb": {"data": "old-rgb"},
        "depth": {"data": "old-depth"},
    }
    new_packet = {
        "seq": 1,
        "_bridge_seq": 2,
        "_transport_session": "new",
        "_transport_connection": 2,
        "stamp": 10.1,
    }
    decoded = _frame(55)

    with patch.object(server_module, "_decode", return_value=decoded):
        old_thread = threading.Thread(
            target=gateway._process_sensor_frame, args=(old_packet,)
        )
        old_thread.start()
        assert entered_commit.wait(1.0)

        transition_finished = threading.Event()

        def transition():
            with gateway._sensor_lock:
                gateway._accept_sensor_generation(new_packet)
            transition_finished.set()

        transition_thread = threading.Thread(target=transition)
        transition_thread.start()
        # A new generation cannot become active in between the old receipt's
        # final check and its RuntimeState/cache commit.
        assert not transition_finished.wait(0.05)
        assert gateway._active_sensor_session == "old"

        release_commit.set()
        old_thread.join(2.0)
        transition_thread.join(2.0)

    assert not old_thread.is_alive()
    assert not transition_thread.is_alive()
    assert gateway.state.frame_seq == 1
    assert gateway._active_sensor_session == "new"


def _invoke_handle_error(server: _ReusableHTTPServer, error: BaseException) -> None:
    try:
        raise error
    except BaseException:
        server.handle_error(object(), ("127.0.0.1", 12345))


def test_http_server_silences_expected_browser_disconnects() -> None:
    server = object.__new__(_ReusableHTTPServer)
    expected = (
        BrokenPipeError(),
        ConnectionResetError(),
        ConnectionAbortedError(),
        socket.timeout(),
        ssl.SSLEOFError(8, "TLS peer closed"),
        OSError(errno.EPIPE, "pipe closed"),
        OSError(errno.ECONNRESET, "peer reset"),
    )
    with patch.object(ThreadingHTTPServer, "handle_error") as parent:
        for error in expected:
            _invoke_handle_error(server, error)
    parent.assert_not_called()
    assert _ReusableHTTPServer.daemon_threads is True


def test_http_server_preserves_unexpected_error_traceback() -> None:
    server = object.__new__(_ReusableHTTPServer)
    with patch.object(ThreadingHTTPServer, "handle_error") as parent:
        _invoke_handle_error(server, ValueError("unexpected"))
    parent.assert_called_once()


def test_recording_submit_worker_retains_only_latest_waiting_batch() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    third_finished = threading.Event()

    class Recorder:
        def __init__(self) -> None:
            self.panels = []
            self.steps = []

        def record_panel(self, index, data, *, stamp=None, frame_seq=None):
            if index == 1 and frame_seq == 1:
                first_started.set()
                assert release_first.wait(2.0)
            self.panels.append((index, data, stamp, frame_seq))
            return True

        def record_step_boundary(self, *, step_index, stamp, frame_seq=None):
            self.steps.append((step_index, stamp, frame_seq))
            if frame_seq == 3:
                third_finished.set()
            return True

    recorder = Recorder()
    worker = _LatestRecordingWorker(recorder)
    assert worker.submit(b"camera-1", {2: b"panel-1"}, b"topology-1",
                         stamp=1.0, frame_seq=1, step_index=1)
    assert first_started.wait(1.0)
    assert worker.submit(b"camera-2", {2: b"panel-2"}, b"topology-2",
                         stamp=2.0, frame_seq=2, step_index=2)
    assert worker.submit(b"camera-3", {2: b"panel-3"}, b"topology-3",
                         stamp=3.0, frame_seq=3, step_index=3)
    release_first.set()
    assert third_finished.wait(2.0)
    worker.close()

    assert [item[2] for item in recorder.steps] == [1, 3]
    assert all(item[3] != 2 for item in recorder.panels)
    assert worker.replaced == 1
