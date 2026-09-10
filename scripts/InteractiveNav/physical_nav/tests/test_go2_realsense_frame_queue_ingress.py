"""Local contract and production-source tests for D435i ``frame_queue`` ingress.

The fake queue keeps mixed-rate behavior executable without ``pyrealsense2``
or a camera. The small reference ingress exercises lifecycle semantics; the
same fake SDK frames also run through the production ``D435iSource.read``.

Contract summary:

* every queue event is demultiplexed once;
* accepted gyro/accelerometer samples are consumed once and in stream order;
* RGB-D delivery has one latest-only pending slot;
* per-stream frame-number gaps, duplicates and late arrivals are observable;
* an RGB-D frame uses the newest pose in the same timestamp domain whose
  timestamp is not newer than the depth exposure;
* ``close()`` wakes a blocked queue wait and joins the ingress thread.
"""

from __future__ import annotations

from bisect import insort
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2_readonly_sensor_bridge import D435iSource, _FrameContinuityTracker

HARDWARE_CLOCK = "hardware_clock"


@dataclass(frozen=True)
class _FakeMotionFrame:
    event_id: str
    stream: str
    frame_number: int
    timestamp_ms: float
    value: tuple[float, float, float]
    timestamp_domain: str = HARDWARE_CLOCK

    def get_frame_number(self) -> int:
        return self.frame_number

    def get_timestamp(self) -> float:
        return self.timestamp_ms

    def get_frame_timestamp_domain(self) -> str:
        return self.timestamp_domain

    def get_profile(self):
        return SimpleNamespace(stream_type=lambda: self.stream)

    def is_motion_frame(self) -> bool:
        return True

    def as_motion_frame(self):
        return self

    def get_motion_data(self):
        return SimpleNamespace(x=self.value[0], y=self.value[1], z=self.value[2])


@dataclass(frozen=True)
class _FakeVideoFrame:
    stream: str
    frame_number: int
    timestamp_ms: float
    timestamp_domain: str
    event_id: str = ""
    metadata: dict[str, float] = field(default_factory=dict)

    def get_frame_number(self) -> int:
        return self.frame_number

    def get_timestamp(self) -> float:
        return self.timestamp_ms

    def get_frame_timestamp_domain(self) -> str:
        return self.timestamp_domain

    def get_profile(self):
        return SimpleNamespace(stream_type=lambda: self.stream)

    def get_data(self):
        if self.stream == "color":
            return np.full((2, 2, 3), self.frame_number, dtype=np.uint8)
        return np.full((2, 2), self.frame_number, dtype=np.uint16)

    def supports_frame_metadata(self, key):
        return key in self.metadata

    def get_frame_metadata(self, key):
        return self.metadata[key]


@dataclass(frozen=True)
class _FakeRgbdFrame:
    event_id: str
    color_frame_number: int
    depth_frame_number: int
    color_timestamp_ms: float
    depth_timestamp_ms: float
    timestamp_domain: str = HARDWARE_CLOCK
    depth_metadata: dict[str, float] = field(default_factory=dict)

    def is_frameset(self) -> bool:
        return True

    def as_frameset(self):
        return self

    def get_color_frame(self) -> _FakeVideoFrame:
        return _FakeVideoFrame(
            "color", self.color_frame_number, self.color_timestamp_ms,
            self.timestamp_domain,
        )

    def get_depth_frame(self) -> _FakeVideoFrame:
        return _FakeVideoFrame(
            "depth", self.depth_frame_number, self.depth_timestamp_ms,
            self.timestamp_domain, metadata=self.depth_metadata,
        )


@dataclass(frozen=True)
class _PoseSnapshot:
    timestamp_ms: float
    timestamp_domain: str
    motion_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class _RgbdSample:
    frame: _FakeRgbdFrame
    pose: _PoseSnapshot | None


class _QueueWoken(RuntimeError):
    pass


class _FakeRealSenseFrameQueue:
    """Blocking fake with the subset needed by a frame-queue ingress."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending = deque()
        self._woken = False
        self.waiting = threading.Event()
        self.delivered_event_ids: list[str] = []
        self.wake_calls = 0

    def enqueue(self, frame: _FakeMotionFrame | _FakeRgbdFrame | _FakeVideoFrame) -> None:
        with self._condition:
            if self._woken:
                raise RuntimeError("queue is closed")
            self._pending.append(frame)
            self._condition.notify()

    def wait_for_frame(
        self, _timeout_ms: int | None = None,
    ) -> _FakeMotionFrame | _FakeRgbdFrame | _FakeVideoFrame:
        with self._condition:
            if not self._pending and not self._woken:
                self.waiting.set()
            self._condition.wait_for(lambda: self._pending or self._woken)
            if self._woken:
                raise _QueueWoken
            frame = self._pending.popleft()
            self.delivered_event_ids.append(frame.event_id)
            return frame

    def poll_for_frame(
        self,
    ) -> _FakeMotionFrame | _FakeRgbdFrame | _FakeVideoFrame | None:
        with self._condition:
            if self._woken:
                raise _QueueWoken
            if not self._pending:
                return None
            frame = self._pending.popleft()
            self.delivered_event_ids.append(frame.event_id)
            return frame

    def size(self) -> int:
        with self._condition:
            return len(self._pending)

    def wake(self) -> None:
        with self._condition:
            self.wake_calls += 1
            self._woken = True
            self._condition.notify_all()


class _ReferenceFrameQueueIngress:
    """Small executable model of the future production ingress contract."""

    def __init__(self, frame_queue: _FakeRealSenseFrameQueue) -> None:
        self._queue = frame_queue
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest_video: _FakeRgbdFrame | None = None
        self._pose_history: list[tuple[float, int, _PoseSnapshot]] = []
        self._pose_insert_order = 0
        self._motion_event_ids: list[str] = []
        self._motion_event_ids_by_domain: dict[str, list[str]] = {}
        self._demuxed_event_ids: list[str] = []
        self._last_frame_number: dict[str, int] = {}
        self._seen_frame_numbers: dict[str, set[int]] = {}
        self._received = Counter()
        self._accepted = Counter()
        self._gaps = Counter()
        self._duplicates = Counter()
        self._late = Counter()
        self._video_replaced = 0
        self._thread = threading.Thread(
            target=self._run, name="test-d435-frame-ingress", daemon=True,
        )
        self._thread.start()

    def _observe_frame_number(self, stream: str, frame_number: int) -> bool:
        self._received[stream] += 1
        previous = self._last_frame_number.get(stream)
        seen = self._seen_frame_numbers.setdefault(stream, set())
        if frame_number in seen:
            self._duplicates[stream] += 1
            return False
        if previous is not None and frame_number < previous:
            self._late[stream] += 1
            return False
        if previous is not None:
            self._gaps[stream] += max(0, frame_number - previous - 1)
        seen.add(frame_number)
        self._last_frame_number[stream] = frame_number
        self._accepted[stream] += 1
        return True

    def _handle_motion(self, frame: _FakeMotionFrame) -> None:
        if not self._observe_frame_number(frame.stream, frame.frame_number):
            return
        self._motion_event_ids.append(frame.event_id)
        domain_events = self._motion_event_ids_by_domain.setdefault(
            frame.timestamp_domain, [],
        )
        domain_events.append(frame.event_id)
        snapshot = _PoseSnapshot(
            timestamp_ms=frame.timestamp_ms,
            timestamp_domain=frame.timestamp_domain,
            motion_event_ids=tuple(domain_events),
        )
        self._pose_insert_order += 1
        insort(
            self._pose_history,
            (frame.timestamp_ms, self._pose_insert_order, snapshot),
        )

    def _handle_rgbd(self, frame: _FakeRgbdFrame) -> None:
        color_ok = self._observe_frame_number("color", frame.color_frame_number)
        depth_ok = self._observe_frame_number("depth", frame.depth_frame_number)
        if not (color_ok and depth_ok):
            return
        if self._latest_video is not None:
            self._video_replaced += 1
        self._latest_video = frame

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._queue.wait_for_frame()
            except _QueueWoken:
                break
            with self._condition:
                self._demuxed_event_ids.append(frame.event_id)
                if isinstance(frame, _FakeMotionFrame):
                    self._handle_motion(frame)
                else:
                    self._handle_rgbd(frame)
                self._condition.notify_all()

    def wait_until_demuxed(self, count: int, timeout: float = 1.0) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: len(self._demuxed_event_ids) >= count,
                timeout=max(0.0, timeout),
            )

    def take_latest_video(self) -> _RgbdSample | None:
        with self._condition:
            frame, self._latest_video = self._latest_video, None
            if frame is None:
                return None
            candidates = (
                (timestamp, order, snapshot)
                for timestamp, order, snapshot in self._pose_history
                if timestamp <= frame.depth_timestamp_ms
                and snapshot.timestamp_domain == frame.timestamp_domain
            )
            latest = max(candidates, key=lambda item: item[:2], default=None)
            pose = latest[2] if latest is not None else None
            return _RgbdSample(frame=frame, pose=pose)

    def stats(self) -> dict[str, object]:
        with self._condition:
            return {
                "demuxed_event_ids": tuple(self._demuxed_event_ids),
                "processed_motion_event_ids": tuple(self._motion_event_ids),
                "received": dict(self._received),
                "accepted": dict(self._accepted),
                "gaps": dict(self._gaps),
                "duplicates": dict(self._duplicates),
                "late": dict(self._late),
                "last_frame_number": dict(self._last_frame_number),
                "video_replaced": self._video_replaced,
                "video_pending": self._latest_video is not None,
                "alive": self._thread.is_alive(),
            }

    def close(self, timeout: float = 1.0) -> bool:
        self._stop.set()
        self._queue.wake()
        if threading.current_thread() is not self._thread:
            self._thread.join(max(0.0, timeout))
        return not self._thread.is_alive()


def _start_ingress():
    queue = _FakeRealSenseFrameQueue()
    ingress = _ReferenceFrameQueueIngress(queue)
    assert queue.waiting.wait(1.0)
    return queue, ingress


def test_mixed_native_rates_consume_motion_once_and_keep_only_latest_video():
    queue, ingress = _start_ingress()
    events: list[_FakeMotionFrame | _FakeRgbdFrame] = []
    expected_motion_ids = []
    for index in range(200):
        event_id = f"gyro-{index + 1}"
        expected_motion_ids.append(event_id)
        events.append(_FakeMotionFrame(
            event_id, "gyro", index + 1, index * 5.0, (0.0, 0.0, 0.01),
        ))
    for index in range(100):
        event_id = f"accel-{index + 1}"
        expected_motion_ids.append(event_id)
        events.append(_FakeMotionFrame(
            event_id, "accel", index + 1, index * 10.0, (0.0, 0.0, 9.81),
        ))
    for index in range(10):
        events.append(_FakeRgbdFrame(
            f"rgbd-{index + 1}", index + 1, index + 1,
            index * 100.0 + 99.5, index * 100.0 + 99.0,
        ))
    events.sort(key=lambda item: (
        item.timestamp_ms if isinstance(item, _FakeMotionFrame)
        else item.depth_timestamp_ms,
        isinstance(item, _FakeRgbdFrame),
    ))

    try:
        for event in events:
            queue.enqueue(event)
        assert ingress.wait_until_demuxed(len(events))
        sample = ingress.take_latest_video()
        stats = ingress.stats()
    finally:
        assert ingress.close()

    assert queue.delivered_event_ids == [event.event_id for event in events]
    assert stats["demuxed_event_ids"] == tuple(queue.delivered_event_ids)
    assert len(stats["processed_motion_event_ids"]) == 300
    assert Counter(stats["processed_motion_event_ids"]) == Counter(expected_motion_ids)
    assert stats["accepted"] == {
        "gyro": 200, "accel": 100, "color": 10, "depth": 10,
    }
    assert stats["video_replaced"] == 9
    assert sample is not None and sample.frame.event_id == "rgbd-10"
    assert sample.pose is not None
    assert sample.pose.timestamp_ms == 995.0
    assert sample.pose.timestamp_ms <= sample.frame.depth_timestamp_ms
    assert set(sample.pose.motion_event_ids) == set(expected_motion_ids)


def test_frame_number_gap_duplicate_and_late_arrival_are_observable_and_dropped():
    queue, ingress = _start_ingress()
    frames = [
        _FakeMotionFrame("g1", "gyro", 1, 5.0, (0.0, 0.0, 0.0)),
        _FakeMotionFrame("g2", "gyro", 2, 10.0, (0.0, 0.0, 0.0)),
        _FakeMotionFrame("g2-duplicate", "gyro", 2, 10.0, (0.0, 0.0, 0.0)),
        _FakeMotionFrame("g5", "gyro", 5, 25.0, (0.0, 0.0, 0.0)),
        _FakeMotionFrame("g4-late", "gyro", 4, 20.0, (0.0, 0.0, 0.0)),
    ]
    try:
        for frame in frames:
            queue.enqueue(frame)
        assert ingress.wait_until_demuxed(len(frames))
        stats = ingress.stats()
    finally:
        assert ingress.close()

    assert stats["demuxed_event_ids"] == tuple(frame.event_id for frame in frames)
    assert stats["processed_motion_event_ids"] == ("g1", "g2", "g5")
    assert stats["received"]["gyro"] == 5
    assert stats["accepted"]["gyro"] == 3
    assert stats["gaps"]["gyro"] == 2
    assert stats["duplicates"]["gyro"] == 1
    assert stats["late"]["gyro"] == 1
    assert stats["last_frame_number"]["gyro"] == 5


def test_depth_exposure_selects_same_domain_pose_not_newer_than_exposure():
    queue, ingress = _start_ingress()
    frames = [
        _FakeMotionFrame("g90", "gyro", 1, 90.0, (0.0, 0.0, 0.1)),
        _FakeMotionFrame("a95", "accel", 1, 95.0, (0.0, 0.0, 9.81)),
        # Both motion samples at this hardware timestamp must be included.
        _FakeMotionFrame("g95", "gyro", 2, 95.0, (0.0, 0.0, 0.1)),
        # Color is deliberately newer: depth is the synchronization anchor.
        _FakeRgbdFrame("rgbd", 1, 1, 120.0, 100.0),
        _FakeMotionFrame("g110", "gyro", 3, 110.0, (0.0, 0.0, 0.1)),
        _FakeMotionFrame(
            "foreign99", "accel", 2, 99.0, (0.0, 0.0, 9.81),
            timestamp_domain="system_time",
        ),
    ]
    try:
        for frame in frames:
            queue.enqueue(frame)
        assert ingress.wait_until_demuxed(len(frames))
        sample = ingress.take_latest_video()
    finally:
        assert ingress.close()

    assert sample is not None and sample.pose is not None
    assert sample.pose.timestamp_domain == HARDWARE_CLOCK
    assert sample.pose.timestamp_ms == 95.0
    assert sample.pose.motion_event_ids == ("g90", "a95", "g95")
    assert "g110" not in sample.pose.motion_event_ids
    assert "foreign99" not in sample.pose.motion_event_ids


def test_close_wakes_an_empty_blocking_queue_and_joins_the_worker():
    queue, ingress = _start_ingress()
    assert ingress.stats()["alive"] is True
    assert ingress.close(timeout=1.0)
    assert queue.wake_calls == 1
    assert ingress.stats()["alive"] is False


def _production_source(queue, *, capacity=512, align_to="none"):
    source = object.__new__(D435iSource)
    source.rs = SimpleNamespace(
        stream=SimpleNamespace(
            gyro="gyro", accel="accel", color="color", depth="depth",
        ),
        frame_metadata_value=SimpleNamespace(
            sensor_timestamp="sensor_timestamp",
            frame_timestamp="frame_timestamp",
            actual_exposure="actual_exposure",
        ),
    )
    source.pipeline = SimpleNamespace()
    source._frame_queue = queue
    source._frame_queue_capacity = capacity
    source.motion_enabled = True
    source.align = None
    source.align_to = align_to
    source.latest_motion = {}
    source._imu_gravity_tau_s = .5
    source._imu_rpy = [0., 0., 0.]
    source._imu_reference_rpy = [0., 0., 0.]
    source._imu_last_gyro_ts = None
    source._imu_last_accel_ts = None
    source._imu_accel_updates = 0
    source._imu_gyro_updates = 0
    source._imu_component_received_at = {"accel": None, "gyro": None}
    source._imu_calibration_samples = 0
    return source


def test_production_source_drains_mixed_rates_once_and_returns_latest_rgbd():
    queue = _FakeRealSenseFrameQueue()
    source = _production_source(queue)
    events: list[_FakeMotionFrame | _FakeRgbdFrame] = []
    for index in range(200):
        events.append(_FakeMotionFrame(
            f"gyro-{index + 1}", "gyro", index + 1, index * 5.,
            (0., 0., .01),
        ))
    for index in range(100):
        events.append(_FakeMotionFrame(
            f"accel-{index + 1}", "accel", index + 1, index * 10.,
            (0., 0., 9.80665),
        ))
    for index in range(10):
        events.append(_FakeRgbdFrame(
            f"rgbd-{index + 1}", index + 1, index + 1,
            index * 100. + 99.5, index * 100. + 99.,
        ))
    events.sort(key=lambda item: (
        item.timestamp_ms if isinstance(item, _FakeMotionFrame)
        else item.depth_timestamp_ms,
        isinstance(item, _FakeRgbdFrame),
    ))
    for event in events:
        queue.enqueue(event)

    rgb, depth, sync_ms = source.read()
    stats = source.ingress_stats()
    motion = source.capture_motion_snapshot()

    assert queue.delivered_event_ids == [event.event_id for event in events]
    assert int(rgb[0, 0, 0]) == 10 and int(depth[0, 0]) == 10
    assert sync_ms == .5
    assert stats["streams"]["gyro"]["dequeued"] == 200
    assert stats["streams"]["gyro"]["unique"] == 200
    assert stats["streams"]["gyro"]["healthy"] is True
    assert stats["streams"]["accel"]["dequeued"] == 100
    assert stats["streams"]["accel"]["unique"] == 100
    assert stats["streams"]["accel"]["healthy"] is True
    assert stats["streams"]["color"]["unique"] == 10
    assert stats["streams"]["depth"]["unique"] == 10
    assert stats["video_replaced"] == 9
    assert stats["queue_dequeued"] == 310
    assert stats["queue_high_water"] == 310
    assert stats["invalid_frames"] == 0
    assert motion["gyro_timestamp_ms"] == 995.
    assert motion["imu_valid_updates"] == {"accel": 100, "gyro": 200}


def test_production_source_reports_gap_duplicate_and_late_motion():
    queue = _FakeRealSenseFrameQueue()
    source = _production_source(queue)
    events = [
        _FakeMotionFrame("g1", "gyro", 1, 5., (0., 0., 0.)),
        _FakeMotionFrame("g2", "gyro", 2, 10., (0., 0., 0.)),
        _FakeMotionFrame("g2-duplicate", "gyro", 2, 10., (0., 0., 0.)),
        _FakeMotionFrame("g5", "gyro", 5, 25., (0., 0., 0.)),
        _FakeMotionFrame("g4-late", "gyro", 4, 20., (0., 0., 0.)),
        _FakeRgbdFrame("rgbd", 1, 1, 30., 30.),
    ]
    for event in events:
        queue.enqueue(event)

    source.read()
    gyro = source.ingress_stats()["streams"]["gyro"]

    assert gyro["dequeued"] == 5
    assert gyro["unique"] == 3
    assert gyro["missing"] == 2
    assert gyro["duplicates"] == 1
    assert gyro["late"] == 1
    assert gyro["last_frame_number"] == 5
    motion = source.capture_motion_snapshot()
    assert motion["frame_number"] == 5
    assert motion["imu_ingress"]["streams"]["gyro"]["healthy"] is False


def test_continuity_requires_progressing_cross_stream_evidence_for_clock_reset():
    tracker = _FrameContinuityTracker()
    for stream in ("color", "depth"):
        assert tracker.observe(
            stream, frame_number=1000, timestamp_ms=10_000.,
            timestamp_domain=HARDWARE_CLOCK,
        )

    # One stale RGB-D composite is not enough to redefine the device epoch.
    assert not tracker.observe(
        "color", frame_number=500, timestamp_ms=5_000.,
        timestamp_domain=HARDWARE_CLOCK,
    )
    assert not tracker.observe(
        "depth", frame_number=500, timestamp_ms=5_000.,
        timestamp_domain=HARDWARE_CLOCK,
    )
    assert tracker.epoch() == 0

    # A second progressing frame in both streams confirms an in-place SDK
    # restart even when the first observed reset frame number is well above 5.
    assert not tracker.observe(
        "color", frame_number=501, timestamp_ms=5_010.,
        timestamp_domain=HARDWARE_CLOCK,
    )
    assert tracker.epoch() == 0
    assert tracker.observe(
        "depth", frame_number=501, timestamp_ms=5_010.,
        timestamp_domain=HARDWARE_CLOCK,
    )
    assert tracker.epoch() == 1
    diagnostics = tracker.diagnostics()
    assert diagnostics["reset_count"] == 1
    assert diagnostics["last_reset_reason"] == "coherent_device_clock_reset"


def test_production_source_recovers_motion_and_video_after_in_place_sdk_reset():
    queue = _FakeRealSenseFrameQueue()
    source = _production_source(queue)
    for event in (
        _FakeMotionFrame("old-g", "gyro", 200, 10_000., (0., 0., .1)),
        _FakeMotionFrame("old-a", "accel", 200, 10_000., (0., 0., 9.80665)),
        _FakeRgbdFrame("old-rgbd", 200, 200, 10_000., 10_000.),
    ):
        queue.enqueue(event)
    source.read()

    for number, timestamp in ((10, 10.), (11, 20.), (12, 30.)):
        queue.enqueue(_FakeMotionFrame(
            f"new-g-{number}", "gyro", number, timestamp, (0., 0., .1),
        ))
        queue.enqueue(_FakeMotionFrame(
            f"new-a-{number}", "accel", number, timestamp,
            (0., 0., 9.80665),
        ))
        queue.enqueue(_FakeRgbdFrame(
            f"new-rgbd-{number}", number, number, timestamp, timestamp,
        ))

    rgb, depth, _sync_ms = source.read()
    stats = source.ingress_stats()

    assert int(rgb[0, 0, 0]) == 12
    assert int(depth[0, 0]) == 12
    assert source._imu_last_gyro_ts == 30.
    assert source._imu_last_accel_ts == 30.
    assert source.latest_motion_snapshot()["frame_number"] == 12
    assert stats["continuity_epoch"]["reset_count"] == 1
    assert stats["device_clock"]["reset_count"] == 1
    assert stats["device_clock"]["last_reset_reason"] == "ingress_continuity_epoch"


def test_late_old_domain_frame_cannot_reset_current_production_epoch():
    source = _production_source(_FakeRealSenseFrameQueue())
    source._process_candidate(_FakeMotionFrame(
        "hardware-g", "gyro", 100, 1000., (0., 0., .1), HARDWARE_CLOCK,
    ))
    source._process_candidate(_FakeMotionFrame(
        "hardware-a", "accel", 100, 1000., (0., 0., 9.80665), HARDWARE_CLOCK,
    ))
    # Two progressing observations in both motion streams establish a real
    # SDK-wide transition; no single foreign sample can clear IMU history.
    for number, timestamp in ((101, 2000.), (102, 2010.)):
        source._process_candidate(_FakeMotionFrame(
            f"global-g-{number}", "gyro", number, timestamp,
            (0., 0., .1), "global_time",
        ))
        source._process_candidate(_FakeMotionFrame(
            f"global-a-{number}", "accel", number, timestamp,
            (0., 0., 9.80665), "global_time",
        ))
    source._process_candidate(_FakeMotionFrame(
        "global-g-head", "gyro", 103, 2020., (0., 0., .1), "global_time",
    ))
    current_epoch = source.ingress_stats()["continuity_epoch"]["epoch"]

    source._process_candidate(_FakeMotionFrame(
        "late-hardware", "gyro", 100, 1000., (0., 0., .1), HARDWARE_CLOCK,
    ))

    assert source.ingress_stats()["continuity_epoch"]["epoch"] == current_epoch
    assert source.latest_motion_snapshot()["timestamp_domain"] == "global_time"
    assert source._imu_last_gyro_ts == 2020.


def test_production_source_uses_same_domain_pose_not_after_depth_exposure():
    queue = _FakeRealSenseFrameQueue()
    source = _production_source(queue)
    events = [
        _FakeMotionFrame("g90", "gyro", 1, 90., (0., 0., .1)),
        _FakeMotionFrame("a95", "accel", 1, 95., (0., 0., 9.80665)),
        _FakeMotionFrame("g95", "gyro", 2, 95., (0., 0., .1)),
        _FakeRgbdFrame("rgbd", 1, 1, 120., 100.),
        _FakeMotionFrame("g110", "gyro", 3, 110., (0., 0., .1)),
        _FakeMotionFrame(
            "foreign99", "accel", 2, 99., (0., 0., 9.80665),
            timestamp_domain="system_time",
        ),
    ]
    for event in events:
        queue.enqueue(event)

    source.read()
    motion = source.capture_motion_snapshot()

    assert motion["timestamp_domain"] == HARDWARE_CLOCK
    assert motion["gyro_timestamp_ms"] == 95.
    assert motion["frame_number"] == 2
    assert motion["gyro_timestamp_ms"] <= 100.


def test_production_source_uses_sensor_exposure_not_later_frame_readout_for_imu():
    queue = _FakeRealSenseFrameQueue()
    source = _production_source(queue)
    events = [
        _FakeMotionFrame("g90", "gyro", 1, 90., (0., 0., .1)),
        _FakeMotionFrame("a95", "accel", 1, 95., (0., 0., 9.80665)),
        _FakeMotionFrame("g95", "gyro", 2, 95., (0., 0., .1)),
        # This sample is before FRAME_TIMESTAMP/readout but after exposure.
        _FakeMotionFrame("g105", "gyro", 3, 105., (0., 0., .1)),
        _FakeRgbdFrame(
            "rgbd", 1, 1, 110., 110., HARDWARE_CLOCK,
            depth_metadata={
                "sensor_timestamp": 100_000.,
                "frame_timestamp": 110_000.,
            },
        ),
    ]
    for event in events:
        queue.enqueue(event)

    source.read()
    motion = source.capture_motion_snapshot()
    timing = source.capture_time_snapshot()

    assert motion["frame_number"] == 2
    assert motion["gyro_timestamp_ms"] == 95.
    assert timing["timestamp_kind"] == "sensor_exposure_midpoint"
    assert timing["imu_reference_timestamp_ms"] == 100.
    assert timing["frame_timestamp_ms"] == 110.


def test_capture_observation_precedes_align_and_uses_raw_depth_identity(monkeypatch):
    import go2_readonly_sensor_bridge as bridge_module

    class _Clock:
        monotonic_value = 10.
        wall_value = 100.

        def monotonic(self):
            return self.monotonic_value

        def wall(self):
            return self.wall_value

        def advance(self, delta):
            self.monotonic_value += delta
            self.wall_value += delta

    class _SlowSyntheticAlign:
        def __init__(self, clock):
            self.clock = clock

        def process(self, _frames):
            self.clock.advance(.08)
            return _FakeRgbdFrame("aligned", 9, 9, 999., 999.)

    clock = _Clock()
    monkeypatch.setattr(bridge_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(bridge_module.time, "time", clock.wall)
    queue = _FakeRealSenseFrameQueue()
    source = _production_source(queue, align_to="color")
    source.align = _SlowSyntheticAlign(clock)
    queue.enqueue(_FakeRgbdFrame("raw", 1, 1, 100., 100.))

    _rgb, depth, _sync_ms = source.read()
    timing = source.capture_time_snapshot()

    assert clock.monotonic_value == pytest.approx(10.08)
    assert timing["dequeue_monotonic"] == 10.
    assert timing["dequeue_wall"] == 100.
    assert timing["sdk_frame_timestamp_ms"] == 100.
    assert timing["frame_timestamp_ms"] == 100.
    assert int(depth[0, 0]) == 9


def test_production_source_pairs_individual_video_frames_and_keeps_latest_pair():
    queue = _FakeRealSenseFrameQueue()
    source = _production_source(queue)
    events: list[_FakeVideoFrame] = []
    for index in range(10):
        number = index + 1
        events.extend((
            _FakeVideoFrame(
                "color", number, index * 100. + 99.5, HARDWARE_CLOCK,
                f"color-{number}",
            ),
            _FakeVideoFrame(
                "depth", number, index * 100. + 99., HARDWARE_CLOCK,
                f"depth-{number}",
            ),
        ))
    for event in events:
        queue.enqueue(event)

    rgb, depth, sync_ms = source.read()
    stats = source.ingress_stats()

    assert queue.delivered_event_ids == [event.event_id for event in events]
    assert int(rgb[0, 0, 0]) == 10
    assert int(depth[0, 0]) == 10
    assert sync_ms == .5
    assert stats["streams"]["color"]["unique"] == 10
    assert stats["streams"]["depth"]["unique"] == 10
    assert stats["video_replaced"] == 9
    assert stats["unpaired_video_frames"] == 0
    assert stats["invalid_frames"] == 0


def test_production_source_discards_only_older_unpairable_video_frame():
    queue = _FakeRealSenseFrameQueue()
    source = _production_source(queue)
    events = [
        _FakeVideoFrame("color", 1, 0., HARDWARE_CLOCK, "color-1"),
        # Outside the 50 ms pairing window, so color-1 is now an orphan.
        _FakeVideoFrame("depth", 1, 80., HARDWARE_CLOCK, "depth-1"),
        _FakeVideoFrame("color", 2, 100., HARDWARE_CLOCK, "color-2"),
    ]
    for event in events:
        queue.enqueue(event)

    rgb, depth, sync_ms = source.read()
    stats = source.ingress_stats()

    assert int(rgb[0, 0, 0]) == 2
    assert int(depth[0, 0]) == 1
    assert sync_ms == 20.
    assert stats["unpaired_video_frames"] == 1
    assert stats["video_pair_skew_exceeded"] == 1
    assert stats["video_pair_domain_mismatches"] == 0
    assert stats["invalid_frames"] == 0


def test_individual_pairing_uses_rate_dependent_window_and_latest_frames():
    source = _production_source(_FakeRealSenseFrameQueue())
    source._video_pair_max_delta_ms = 500. / 30.
    frames = [
        _FakeVideoFrame("color", 1, 0., HARDWARE_CLOCK, "color-1"),
        _FakeVideoFrame("color", 2, 33.3, HARDWARE_CLOCK, "color-2"),
        _FakeVideoFrame("depth", 1, .5, HARDWARE_CLOCK, "depth-1"),
        _FakeVideoFrame("depth", 2, 33.8, HARDWARE_CLOCK, "depth-2"),
    ]

    pairs = [source._process_candidate(frame) for frame in frames]
    pair = pairs[-1]
    stats = source.ingress_stats()

    assert pair is not None
    assert pair.get_color_frame().frame_number == 2
    assert pair.get_depth_frame().frame_number == 2
    assert all(candidate is None for candidate in pairs[:-1])
    assert stats["unpaired_video_frames"] == 2
    assert stats["video_pair_skew_exceeded"] == 1


def test_individual_pairing_rejects_unknown_or_different_timestamp_domains():
    source = _production_source(_FakeRealSenseFrameQueue())
    frames = [
        _FakeVideoFrame("color", 1, 0., "", "color-unknown"),
        _FakeVideoFrame("depth", 1, 1., HARDWARE_CLOCK, "depth-1"),
        _FakeVideoFrame("color", 2, 2., HARDWARE_CLOCK, "color-2"),
    ]

    pairs = [source._process_candidate(frame) for frame in frames]
    stats = source.ingress_stats()

    assert pairs[0] is None and pairs[1] is None
    assert pairs[2] is not None
    assert stats["video_pair_domain_mismatches"] == 1
    assert stats["unpaired_video_frames"] == 1


def test_individual_pairing_rejects_matching_but_unknown_timestamp_domains():
    source = _production_source(_FakeRealSenseFrameQueue())

    first = source._process_candidate(
        _FakeVideoFrame("color", 1, 10., "bogus_clock", "color-bogus")
    )
    second = source._process_candidate(
        _FakeVideoFrame("depth", 1, 10., "bogus_clock", "depth-bogus")
    )

    assert first is None and second is None
    assert source.ingress_stats()["video_pair_domain_mismatches"] == 1


def test_native_frameset_purges_pending_individual_video():
    source = _production_source(_FakeRealSenseFrameQueue())
    assert source._process_candidate(
        _FakeVideoFrame("color", 1, 10., HARDWARE_CLOCK, "color-1")
    ) is None

    frameset = source._process_candidate(
        _FakeRgbdFrame("rgbd-2", 2, 2, 20., 20.)
    )
    stats = source.ingress_stats()

    assert frameset is not None
    assert source._pending_video_frames == {"color": None, "depth": None}
    assert stats["unpaired_video_frames"] == 1


def test_align_color_rejects_duck_typed_individual_video_pair():
    source = _production_source(_FakeRealSenseFrameQueue(), align_to="color")

    assert source._process_candidate(
        _FakeVideoFrame("color", 1, 10., HARDWARE_CLOCK, "color-1")
    ) is None
    assert source._process_candidate(
        _FakeVideoFrame("depth", 1, 10., HARDWARE_CLOCK, "depth-1")
    ) is None

    stats = source.ingress_stats()
    assert stats["unsupported_align_color_video_pairs"] == 1
    assert source._pending_video_frames == {"color": None, "depth": None}


def test_production_source_close_wakes_blocked_read_and_is_idempotent():
    queue = _FakeRealSenseFrameQueue()
    source = _production_source(queue)
    stop_calls = []
    source.pipeline = SimpleNamespace(
        stop=lambda: (stop_calls.append(True), queue.wake()),
    )
    errors: list[Exception] = []

    def read_until_closed():
        try:
            source.read()
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=read_until_closed, daemon=True)
    worker.start()
    assert queue.waiting.wait(1.)

    source.close()
    source.close()
    worker.join(1.)

    assert not worker.is_alive()
    assert len(stop_calls) == 1
    assert queue.wake_calls == 1
    assert len(errors) == 1
    assert "closed" in str(errors[0]).lower()


def test_production_source_close_can_retry_transient_pipeline_stop_failure():
    source = _production_source(_FakeRealSenseFrameQueue())
    stop_calls = []

    def stop():
        stop_calls.append(True)
        if len(stop_calls) == 1:
            raise RuntimeError("temporary stop failure")

    source.pipeline = SimpleNamespace(stop=stop)

    with pytest.raises(RuntimeError, match="temporary stop failure"):
        source.close()
    assert source._closed_event.is_set()
    assert source._pipeline_stopped is False

    source.close()
    source.close()
    assert len(stop_calls) == 2
    assert source._pipeline_stopped is True
