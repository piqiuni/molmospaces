"""Deterministic regressions for the latest-only HTTP grid mirror."""

from __future__ import annotations

import json as std_json
import pathlib
import sys
import threading
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import physical_ros_gateway as gateway_module
from physical_ros_gateway import PhysicalRosGateway


def _gateway(*, occupancy_period: float = 0.0, room_period: float = 0.0):
    gateway = object.__new__(PhysicalRosGateway)
    gateway.web_state_enabled = True
    gateway.args = SimpleNamespace(
        occupancy_period=occupancy_period,
        room_grid_period=room_period,
        web_url="http://127.0.0.1:8765",
    )
    gateway._web_session_id = "test-session"
    gateway._last_grid_post = {}
    gateway._grid_post_lock = threading.Lock()
    gateway._pending_grid_posts = {}
    gateway._grid_post_revisions = {}
    gateway._grid_post_event = threading.Event()
    return gateway


def _message(data=(0,)):
    origin = SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="tf_frame_map"),
        info=SimpleNamespace(width=len(data), height=1, resolution=0.05, origin=origin),
        data=data,
    )


def test_callback_keeps_only_newest_message_without_materializing_data():
    gateway = _gateway(occupancy_period=0.2)

    class MustNotIterate:
        def __iter__(self):
            raise AssertionError("ROS callback must not materialize grid data")

        def __len__(self):
            return 1

    first = _message(MustNotIterate())
    second = _message(MustNotIterate())

    gateway._grid_callback("occupancy")(first)
    gateway._grid_callback("occupancy")(second)

    assert gateway._grid_post_revisions == {"occupancy": 2}
    assert gateway._pending_grid_posts == {"occupancy": (2, second)}
    # The callback no longer starts the presentation rate-limit window.
    assert gateway._last_grid_post == {}


def test_worker_claims_occ_first_and_other_slots_remain_replaceable():
    gateway = _gateway()
    old_global = _message((1,))
    occupancy = _message((2,))
    new_global = _message((3,))
    gateway._grid_callback("global_costmap")(old_global)
    gateway._grid_callback("occupancy")(occupancy)

    claimed, _ = gateway._claim_next_grid_post(now=10.0)
    assert claimed == ("occupancy", 1, occupancy)
    assert gateway._pending_grid_posts["global_costmap"] == (1, old_global)

    # This arrives while OCC is conceptually in-flight. Because the worker
    # claimed one item rather than swapping the whole dict, it replaces the
    # unprocessed global map instead of waiting behind a frozen old receipt.
    gateway._grid_callback("global_costmap")(new_global)
    claimed, _ = gateway._claim_next_grid_post(now=10.0)
    assert claimed == ("global_costmap", 2, new_global)


def test_continuous_occupancy_does_not_starve_other_due_grids():
    gateway = _gateway()
    first_occ = _message((1,))
    room = _message((2,))
    next_occ = _message((3,))
    gateway._grid_callback("room_grid")(room)
    gateway._grid_callback("occupancy")(first_occ)

    claimed, _ = gateway._claim_next_grid_post(now=10.0)
    assert claimed == ("occupancy", 1, first_occ)

    # Simulate another OCC receipt arriving while the first HTTP request is
    # in flight. The already-waiting room grid gets one turn before OCC can be
    # selected again, even if both remain due.
    gateway._grid_callback("occupancy")(next_occ)
    claimed, _ = gateway._claim_next_grid_post(now=10.1)
    assert claimed == ("room_grid", 1, room)
    claimed, _ = gateway._claim_next_grid_post(now=10.2)
    assert claimed == ("occupancy", 2, next_occ)


def test_worker_rate_limit_retains_the_latest_frame_in_the_period():
    gateway = _gateway(occupancy_period=0.2)
    first = _message((1,))
    middle = _message((2,))
    latest = _message((3,))
    gateway._grid_callback("occupancy")(first)
    claimed, _ = gateway._claim_next_grid_post(now=10.0)
    assert claimed == ("occupancy", 1, first)

    gateway._grid_callback("occupancy")(middle)
    claimed, delay = gateway._claim_next_grid_post(now=10.1)
    assert claimed is None
    assert 0.09 <= delay <= 0.11
    gateway._grid_callback("occupancy")(latest)

    claimed, _ = gateway._claim_next_grid_post(now=10.21)
    assert claimed == ("occupancy", 3, latest)


def test_obsolete_revision_is_dropped_before_large_data_copy():
    gateway = _gateway()
    gateway._grid_post_revisions["occupancy"] = 2

    class MustNotIterate:
        def __iter__(self):
            raise AssertionError("obsolete grid must be dropped before list(data)")

    gateway._post_state_http = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("obsolete grid must not reach JSON/HTTP")
    )

    assert not gateway._post_grid_state(
        "occupancy", SimpleNamespace(data=MustNotIterate()), revision=1
    )


def test_replacement_during_data_copy_cancels_json_and_http():
    gateway = _gateway()
    replacement = _message((9,))

    class ReplacingData:
        def __len__(self):
            return 2

        def __iter__(self):
            gateway._grid_callback("occupancy")(replacement)
            return iter((-1, 0))

    old = _message(ReplacingData())
    gateway._grid_callback("occupancy")(old)
    claimed, _ = gateway._claim_next_grid_post(now=10.0)
    assert claimed == ("occupancy", 1, old)
    gateway._post_state_http = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("superseded copied grid must not reach JSON/HTTP")
    )

    assert not gateway._post_grid_state("occupancy", old, revision=1)
    assert gateway._pending_grid_posts["occupancy"] == (2, replacement)


def test_replacement_during_json_cancels_before_http(monkeypatch):
    gateway = _gateway()
    gateway._grid_post_revisions["occupancy"] = 1
    real_dumps = std_json.dumps

    def replacing_dumps(*args, **kwargs):
        encoded = real_dumps(*args, **kwargs)
        gateway._grid_callback("occupancy")(_message((7,)))
        return encoded

    monkeypatch.setattr(gateway_module.json, "dumps", replacing_dumps)
    monkeypatch.setattr(
        gateway_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("superseded JSON must not be sent")
        ),
    )

    assert not gateway._post_state_http(
        "occupancy",
        {"data": [0]},
        is_current=lambda: gateway._grid_revision_is_current("occupancy", 1),
    )


def test_replacement_during_request_build_cancels_before_socket(monkeypatch):
    gateway = _gateway()
    gateway._grid_post_revisions["occupancy"] = 1

    def replacing_request(*_args, **_kwargs):
        gateway._grid_callback("occupancy")(_message((8,)))
        return object()

    monkeypatch.setattr(gateway_module.urllib.request, "Request", replacing_request)
    monkeypatch.setattr(
        gateway_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("superseded request must not open a socket")
        ),
    )

    assert not gateway._post_state_http(
        "occupancy",
        {"data": [0]},
        is_current=lambda: gateway._grid_revision_is_current("occupancy", 1),
    )
