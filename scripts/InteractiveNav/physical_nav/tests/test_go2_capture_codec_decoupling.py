"""Latest-only codec scheduling must be bounded and independently stoppable."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import go2_readonly_sensor_bridge as bridge
from go2_readonly_sensor_bridge import _LatestOnlyStage
from physical_protocol import decode_wire_packet


def test_blocked_codec_does_not_block_capture_and_keeps_only_latest_pending_frame():
    entered = threading.Event()
    release = threading.Event()
    producer_done = threading.Event()
    second_output = threading.Event()
    processed, outputs = [], []
    accepted = []

    def encode(frame):
        processed.append(frame["seq"])
        if frame["seq"] == 1:
            entered.set()
            assert release.wait(1.)
        return dict(frame)

    def publish(frame):
        outputs.append(frame["seq"])
        if len(outputs) == 2:
            second_output.set()

    stage = _LatestOnlyStage(encode, publish, name="test-latest-codec")
    producer = None
    try:
        assert stage.submit({"seq": 1})
        assert entered.wait(1.)

        def produce():
            for seq in range(2, 7):
                accepted.append(stage.submit({"seq": seq}))
            producer_done.set()

        producer = threading.Thread(target=produce)
        producer.start()
        assert producer_done.wait(1.)
        assert accepted == [True] * 5
        assert processed == [1] and outputs == []
        release.set()
        assert second_output.wait(1.)
        assert processed == [1, 6]
        assert outputs == [1, 6]
    finally:
        release.set()
        if producer is not None:
            producer.join(1.)
        assert stage.close(timeout=1.)
    assert stage.stats() == {
        "submitted": 6, "replaced": 4, "processed": 2,
        "errors": 0, "alive": False,
    }
    assert stage.submit({"seq": 7}) is False


def test_codec_failure_is_counted_and_does_not_kill_worker():
    failed = threading.Event()
    recovered = threading.Event()
    outputs = []

    def encode(frame):
        if frame["seq"] == 1:
            raise ValueError("bad frame")
        return frame

    def publish(frame):
        outputs.append(frame["seq"])
        recovered.set()

    stage = _LatestOnlyStage(
        encode, publish, name="test-codec-recovery",
        on_error=lambda _exc: failed.set(),
    )
    try:
        assert stage.submit({"seq": 1})
        assert failed.wait(1.)
        assert stage.submit({"seq": 2})
        assert recovered.wait(1.)
        assert outputs == [2]
    finally:
        assert stage.close(timeout=1.)
    stats = stage.stats()
    assert stats["errors"] == 1 and stats["processed"] == 1
    assert stats["alive"] is False


def _publish_args(**overrides):
    values = dict(
        interface="unused", dry_run=False,
        depth_width=2, depth_height=2, depth_fps=10,
        color_width=2, color_height=2, color_fps=10,
        enable_camera_imu=False, align_to="depth", imu_gravity_tau_s=.5,
        width=2, height=2, fps=100, depth_scale=.001,
        depth_png_compression=1, url="ws://test", connect_timeout=.1,
        send_timeout=.1, send_buffer_kb=32, camera_frame="camera",
        max_frame_age_s=1., telemetry_period=.001, publish_fps=100.,
        reconnect_s=.001,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeSocket:
    def setsockopt(self, *_args):
        pass


class _FakeWebSocket:
    def __init__(self, *, fail_second_text=False):
        self.sock = _FakeSocket()
        self.fail_second_text = fail_second_text
        self.text_sends = 0
        self.text = []
        self.binary = []
        self.binary_ready = threading.Event()
        self.hello_ready = threading.Event()
        self.closed = 0

    def settimeout(self, _timeout):
        pass

    def send(self, payload):
        self.text_sends += 1
        packet_type = None
        try:
            packet = json.loads(payload)
            packet_type = packet.get("type")
            self.text.append(packet)
        except (AttributeError, TypeError, ValueError):
            pass
        # The production sender now emits hello + two startup clock probes
        # before its first RGB-D payload.  Keep this failure injection tied to
        # the post-frame telemetry send that the reconnect test intends to
        # exercise, rather than to an incidental text-message ordinal.
        if self.fail_second_text and packet_type == "telemetry":
            raise RuntimeError("force reconnect after first frame")
        self.hello_ready.set()
        return len(payload)

    def send_binary(self, payload):
        self.binary.append(payload)
        self.binary_ready.set()

    def close(self):
        self.closed += 1


class _FakeSource:
    instances = []

    def __init__(self, *_args, **_kwargs):
        self.intrinsics = {"fx": 1., "fy": 1., "cx": 1., "cy": 1.,
                           "width": 2, "height": 2}
        self.depth_intrinsics = dict(self.intrinsics)
        self.depth_to_color_extrinsics = {}
        self.depth_scale = .001
        self.latest_motion = {}
        self.closed = threading.Event()
        self.close_count = 0
        self.read_count = 0
        type(self).instances.append(self)

    def read(self):
        if self.closed.is_set():
            raise RuntimeError("closed")
        self.read_count += 1
        time.sleep(.002)
        rgb = np.full((2, 2, 3), self.read_count % 255, dtype=np.uint8)
        depth = np.full((2, 2), self.read_count, dtype=np.uint16)
        return rgb, depth, 0.

    def close(self):
        self.close_count += 1
        self.closed.set()

    def capture_motion_snapshot(self):
        return dict(self.latest_motion)

    def latest_motion_snapshot(self):
        return dict(self.latest_motion)

    def ingress_stats(self):
        return {}


def _install_publish_fakes(monkeypatch, sockets, *, encode_delay_s=0.):
    _FakeSource.instances.clear()
    monkeypatch.setattr(bridge, "D435iSource", _FakeSource)
    monkeypatch.setattr(bridge, "_unitree_state_reader", lambda *_args: None)

    def encode(rgb, depth, **_kwargs):
        if encode_delay_s:
            time.sleep(encode_delay_s)
        return bytes([int(rgb[0, 0, 0])]), bytes([int(depth[0, 0]) % 255])

    monkeypatch.setattr(bridge, "_encode_parallel", encode)
    socket_iter = iter(sockets)
    fake_websocket = SimpleNamespace(
        create_connection=lambda *_args, **_kwargs: next(socket_iter),
    )
    monkeypatch.setitem(sys.modules, "websocket", fake_websocket)


async def _wait_event(event, timeout=1.):
    deadline = time.monotonic() + timeout
    while not event.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(.002)
    return event.is_set()


async def _cancel_after(event, task):
    assert await _wait_event(event)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_publish_cancellation_closes_source_and_joins_owned_workers(monkeypatch):
    ws = _FakeWebSocket()
    _install_publish_fakes(monkeypatch, [ws])

    async def scenario():
        task = asyncio.create_task(bridge.publish(_publish_args()))
        await _cancel_after(ws.binary_ready, task)

    asyncio.run(scenario())
    source = _FakeSource.instances[0]
    assert source.close_count == 1
    assert not any(
        thread.is_alive() and (
            thread.name in {"d435i-capture", "d435i-codec"}
            or thread.name.startswith("go2-rgbd-encode")
        )
        for thread in threading.enumerate()
    )
    assert ws.closed == 1
    assert [packet["type"] for packet in ws.text[:3]] == [
        "hello", "clock_probe", "clock_probe",
    ]


def test_frames_older_than_send_budget_are_never_transmitted(monkeypatch):
    ws = _FakeWebSocket()
    _install_publish_fakes(monkeypatch, [ws], encode_delay_s=.02)

    async def scenario():
        task = asyncio.create_task(bridge.publish(
            _publish_args(max_frame_age_s=.005)
        ))
        assert await _wait_event(ws.hello_ready)
        await asyncio.sleep(.08)
        assert ws.binary == []
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_capture_copies_reused_sdk_buffers_before_async_encoding(monkeypatch):
    ws = _FakeWebSocket()
    encoder_entered = threading.Event()
    release_encoder = threading.Event()
    source_advanced = threading.Event()
    observed_values = []

    class ReusedBufferSource(_FakeSource):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.rgb = np.zeros((2, 2, 3), dtype=np.uint8)
            self.depth = np.zeros((2, 2), dtype=np.uint16)

        def read(self):
            if self.closed.is_set():
                raise RuntimeError("closed")
            self.read_count += 1
            self.rgb.fill(self.read_count)
            self.depth.fill(self.read_count)
            if self.read_count >= 5:
                source_advanced.set()
            time.sleep(.001)
            return self.rgb, self.depth, 0.

    def blocked_encode(rgb, depth, **_kwargs):
        before = (int(rgb[0, 0, 0]), int(depth[0, 0]))
        if not encoder_entered.is_set():
            encoder_entered.set()
            assert release_encoder.wait(1.)
            observed_values.append((before, (int(rgb[0, 0, 0]), int(depth[0, 0]))))
        return b"rgb", b"depth"

    monkeypatch.setattr(bridge, "D435iSource", ReusedBufferSource)
    monkeypatch.setattr(bridge, "_unitree_state_reader", lambda *_args: None)
    monkeypatch.setattr(bridge, "_encode_parallel", blocked_encode)
    monkeypatch.setitem(
        sys.modules, "websocket",
        SimpleNamespace(create_connection=lambda *_args, **_kwargs: ws),
    )

    async def scenario():
        task = asyncio.create_task(bridge.publish(_publish_args()))
        try:
            assert await _wait_event(encoder_entered)
            assert await _wait_event(source_advanced)
            release_encoder.set()
            assert await _wait_event(ws.binary_ready)
        finally:
            release_encoder.set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())
    assert observed_values == [((1, 1), (1, 1))]


def test_reconnect_does_not_resend_the_last_successful_frame(monkeypatch):
    first = _FakeWebSocket(fail_second_text=True)
    second = _FakeWebSocket()

    class OneFrameSource(_FakeSource):
        def read(self):
            if self.read_count:
                time.sleep(.005)
                raise RuntimeError("no newer frame")
            return super().read()

    monkeypatch.setattr(bridge, "D435iSource", OneFrameSource)
    monkeypatch.setattr(bridge, "_unitree_state_reader", lambda *_args: None)
    monkeypatch.setattr(
        bridge, "_encode_parallel",
        lambda rgb, depth, **_kwargs: (b"rgb", b"depth"),
    )
    sockets = iter([first, second])
    monkeypatch.setitem(
        sys.modules, "websocket",
        SimpleNamespace(create_connection=lambda *_args, **_kwargs: next(sockets)),
    )

    async def scenario():
        task = asyncio.create_task(bridge.publish(_publish_args()))
        assert await _wait_event(first.binary_ready)
        assert await _wait_event(second.hello_ready)
        await asyncio.sleep(.03)
        assert len(first.binary) == 1
        assert second.binary == []
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_wire_stamp_and_stale_clock_use_device_mapped_capture_time(monkeypatch):
    ws = _FakeWebSocket()

    class TimedSource(_FakeSource):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.capture_time = {}

        def read(self):
            frame = super().read()
            observed_wall = time.time()
            observed_monotonic = time.monotonic()
            self.capture_time = {
                "valid": True,
                "capture_wall": observed_wall - .02,
                "capture_monotonic": observed_monotonic - .02,
                "dequeue_wall": observed_wall,
                "dequeue_monotonic": observed_monotonic,
                "dequeue_age_s": .02,
                "device_timestamp_ms": float(self.read_count),
                "timestamp_domain": "hardware_clock",
            }
            return frame

        def capture_time_snapshot(self):
            return dict(self.capture_time)

    monkeypatch.setattr(bridge, "D435iSource", TimedSource)
    monkeypatch.setattr(bridge, "_unitree_state_reader", lambda *_args: None)
    monkeypatch.setattr(
        bridge, "_encode_parallel",
        lambda rgb, depth, **_kwargs: (b"rgb", b"depth"),
    )
    monkeypatch.setitem(
        sys.modules, "websocket",
        SimpleNamespace(create_connection=lambda *_args, **_kwargs: ws),
    )

    async def scenario():
        task = asyncio.create_task(bridge.publish(_publish_args()))
        try:
            assert await _wait_event(ws.binary_ready)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())
    packet = decode_wire_packet(ws.binary[0])

    assert packet["stamp"] == pytest.approx(
        packet["capture_timing"]["capture_wall"]
    )
    assert packet["capture_timing"]["stamp_source"] == "device_clock_estimated"
    assert packet["capture_timing"]["dequeue_age_s"] == pytest.approx(.02)
    assert packet["capture_timing"]["selected_capture_wall"] == pytest.approx(
        packet["stamp"]
    )
