"""Exercise explicit stops through the real velocity mux callbacks offline."""

import pathlib
import sys
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("rospy")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import velocity_command_mux as module


def test_explicit_stop_overrides_navigation_without_semantic_lease(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(module.rospy.Time, "now", staticmethod(
        lambda: module.rospy.Time.from_sec(clock[0])))
    monkeypatch.setattr(module.rospy, "loginfo", lambda *args: None)
    mux = object.__new__(module.VelocityCommandMux)
    mux._lock = threading.Lock()
    mux.semantic_timeout_s = 0.40
    mux.move_base_timeout_s = mux.semantic_stop_hold_s = 0.35
    mux._move_base = Twist()
    mux._semantic = Twist()
    for key in ("_move_base_at", "_semantic_at", "_semantic_active_until",
                "_semantic_stop_until", "_explicit_stop_until"):
        setattr(mux, key, module.rospy.Time(0))
    mux._last_source = "none"
    output = []
    mux.publisher = SimpleNamespace(publish=output.append)
    forward = Twist()
    forward.linear.x = 0.4
    mux._move_base_callback(forward)
    mux._semantic_callback(Twist())  # Ordinary idle zeros retain their old semantics.
    mux._publish(None)
    assert output[-1].linear.x == 0.4

    mux._stop_callback(String(data="stop"))
    mux._publish(None)
    assert module._is_zero(output[-1])
    clock[0] = 10.1
    mux._move_base_callback(forward)
    turn = Twist()
    turn.angular.z = 0.5
    mux._semantic_callback(turn)
    mux._publish(None)
    assert module._is_zero(output[-1])  # Neither lane can break the stop hold.

    clock[0] = 10.36
    mux._publish(None)
    assert module._is_zero(output[-1])  # Never replay a cached predecessor command.
    clock[0] = 10.4
    mux._move_base_callback(forward)
    mux._publish(None)
    assert output[-1].linear.x == 0.4  # A fresh successor command can resume.
