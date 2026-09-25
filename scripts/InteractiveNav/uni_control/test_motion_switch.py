import os
import pathlib
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:
    import websocket
except ModuleNotFoundError:
    # These tests use the owner-only Unix socket, not the optional WS client.
    sys.modules["websocket"] = SimpleNamespace()
from control_protocol import ControlCommand
from go2_control_bridge import Go2Driver, MotionController, UnifiedBridge
from motion_switch import MotionSwitchServer, request_switch


class Driver:
    def __init__(self):
        self.moves = []
        self.switches = []
        self.fail = False

    def move(self, *velocity):
        self.moves.append(velocity)

    def set_motion_enabled(self, enabled):
        self.switches.append(enabled)
        if self.fail:
            raise RuntimeError("SDK unavailable")


def bridge():
    instance = object.__new__(UnifiedBridge)
    instance.args = SimpleNamespace(enable_motion=False, state_source="go2",
                                    max_vx=0.6, max_wz=1.3, control_period=0.002)
    instance.driver = Driver()
    instance.controller = MotionController(instance.args, instance.driver, lambda event: None)
    return instance


def command(seq=1):
    return ControlCommand(seq=seq, control_mode="continuous", ttl_ms=500, vx=0.3)


def test_toggle_preserves_bridge_and_requires_new_command():
    b = bridge()
    assert not b.controller.submit(command())
    assert b.switch_motion({"enabled": True})["mode"] == "enable_motion"
    assert b.controller.target(time.monotonic()) == (0, 0, 0)
    assert b.controller.submit(command(2))
    assert b.controller.target(time.monotonic())[0] == 0.3
    b.switch_motion({"enabled": False})
    assert b.driver.moves[-1] == (0, 0, 0)
    assert not b.controller.submit(command(3))
    b.switch_motion({"enabled": True})
    assert b.controller.target(time.monotonic()) == (0, 0, 0)
    calls = list(b.driver.switches)
    b.switch_motion({"enabled": True})
    assert b.driver.switches == calls


def test_failed_enable_and_invalid_request_cannot_open_gate():
    b = bridge()
    for payload in ({"enabled": "true"}, {"enabled": True, "max_vx": float("nan")}):
        with pytest.raises(ValueError):
            b.switch_motion(payload)
    assert not b.driver.switches
    b.driver.fail = True
    with pytest.raises(RuntimeError, match="SDK unavailable"):
        b.switch_motion({"enabled": True})
    assert not b.controller.motion_enabled
    assert not b.args.enable_motion
    assert not b.controller.submit(command())


def test_enable_in_progress_does_not_queue_commands():
    b = bridge()
    entered, release = threading.Event(), threading.Event()
    def blocked_enable(enabled):
        entered.set()
        assert release.wait(2)
    b.driver.set_motion_enabled = blocked_enable
    worker = threading.Thread(target=lambda: b.switch_motion({"enabled": True}))
    worker.start()
    try:
        assert entered.wait(1)
        start = time.monotonic()
        for seq in range(100):
            assert not b.controller.submit(command(seq))
        assert time.monotonic() - start < 0.1
    finally:
        release.set()
        worker.join(2)
    assert b.controller.target(time.monotonic()) == (0, 0, 0)
    assert b.controller.submit(command(101))


def test_disable_waits_for_inflight_move_and_no_old_target_follows():
    b = bridge()
    b.switch_motion({"enabled": True})
    entered, release, disabled = threading.Event(), threading.Event(), threading.Event()
    def move(*velocity):
        if any(velocity):
            entered.set()
            assert release.wait(2)
        b.driver.moves.append(velocity)
    b.driver.move = move
    b.controller.submit(command())
    runner = threading.Thread(target=b.controller.run)
    runner.start()
    changer = threading.Thread(target=lambda: (b.switch_motion({"enabled": False}), disabled.set()))
    try:
        assert entered.wait(1)
        changer.start()
        assert not disabled.wait(0.02)
        release.set()
        assert disabled.wait(1)
        assert b.driver.moves[-1] == (0, 0, 0)
        assert not b.controller.submit(command(2))
    finally:
        release.set()
        b.controller.stop_event.set()
        runner.join(2)
        if changer.ident is not None:
            changer.join(2)
    assert b.driver.moves[-1] == (0, 0, 0)


def test_driver_disable_blocks_lease_auto_reacquisition():
    d = object.__new__(Go2Driver)
    moves, leases = [], []
    d._motion_lock = threading.RLock()
    d._client = SimpleNamespace(Move=lambda *v: moves.append(v),
                                UseRemoteCommandFromApi=lambda v: leases.append(v))
    d._motion_authorized = True
    d.set_motion_enabled(False)
    d.move(0.5, 0, 0)
    assert moves == [(0, 0, 0)]
    assert leases == [False]


def test_owner_socket_roundtrip_retains_process_and_checks_errors(tmp_path):
    b = bridge()
    ready = tmp_path / "ready"
    ready.write_text(str(os.getpid()))
    server = MotionSwitchServer(ready, b.switch_motion)
    try:
        assert server.path.stat().st_mode & 0o777 == 0o600
        assert request_switch(ready, {})["mode"] == "speech_only"
        assert request_switch(ready, {"enabled": True})["pid"] == os.getpid()
        assert request_switch(ready, {"enabled": False})["mode"] == "speech_only"
        with pytest.raises(RuntimeError):
            request_switch(ready, {"enabled": "yes"})
        b.controller.stop_event.set()
        with pytest.raises(RuntimeError, match="stopping"):
            request_switch(ready, {"enabled": True})
    finally:
        server.close()
