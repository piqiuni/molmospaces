import math
import ast
import pathlib
import sys
import threading
from collections import deque

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physical_six_panel_server import SixPanelRenderer, _grid_origin_yaw
from runtime_state import RuntimeState, telemetry_yaw


def test_small_state_post_lane_is_not_throttled_to_two_hz():
    """Keep the ROS-free dashboard state cadence separate from OCC uploads.

    ``physical_ros_gateway`` imports rospy, so this contract is checked from
    its AST in the lightweight test environment.  The small-state worker must
    wake at <=100 ms; the dedicated grid worker keeps a 500 ms base wake-up
    (it may wake earlier when a map post is pending, but never faster than
    the capped 0.5 s timeout) because it only handles latest-only,
    potentially megabyte-sized maps.
    """

    source_path = ROOT / "physical_ros_gateway.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    constants = {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, (int, float))
    }
    assert constants["_STATE_POST_WAIT_S"] <= 0.1
    state_loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_state_post_loop"
    )
    waits = [
        call
        for call in ast.walk(state_loop)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "wait"
    ]
    assert any(
        call.args
        and isinstance(call.args[0], ast.Call)
        and isinstance(call.args[0].func, ast.Name)
        and call.args[0].func.id == "_state_post_wait_seconds"
        for call in waits
    )
    grid_loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_grid_post_loop"
    )
    grid_numbers = [
        node.value
        for node in ast.walk(grid_loop)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ]
    assert 0.5 in grid_numbers, "grid lane must keep its 500 ms base wake-up"
    assert any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "min"
        and any(isinstance(arg, ast.Constant) and arg.value == 0.5 for arg in call.args)
        for call in ast.walk(grid_loop)
    ), "adaptive grid wake-up must stay capped at 0.5 s"


def test_presentation_heartbeat_mirror_is_rate_limited_without_touching_telemetry():
    from physical_ros_gateway import _state_mirror_min_period

    assert _state_mirror_min_period("graph") == 1.0
    assert _state_mirror_min_period("consistency") == 1.0
    assert _state_mirror_min_period("explore_status") == 0.5
    assert _state_mirror_min_period("telemetry") == 0.0
    assert _state_mirror_min_period("mapped_detections") == 0.0


def test_telemetry_yaw_falls_back_to_unitree_imu_rpy_and_quaternion():
    assert telemetry_yaw({"imu": {"rpy": [0.1, -0.2, 0.73]}}) == 0.73
    yaw = -0.61
    assert math.isclose(
        telemetry_yaw(
            {"imu": {"quaternion": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]}}
        ),
        yaw,
        abs_tol=1e-6,
    )


def test_runtime_state_materializes_missing_yaw_from_imu():
    state = RuntimeState()
    state.update_topic(
        "telemetry",
        {"received_at": 1.0, "position": [1.0, 2.0, 0.0], "imu": {"rpy": [0.0, 0.0, 1.1]}},
    )
    assert state.snapshot()["telemetry"]["yaw"] == 1.1


def test_capture_sequence_keeps_heading_live_across_source_clock_rollback():
    """A newer RGB-D frame must not be rejected after a Go2 clock step."""
    state = RuntimeState()
    state.update_frame(
        frame_seq=10,
        frame_stamp=1000.0,
        telemetry={
            "received_at": 1000.0,
            "position": [0.0, 0.0, 0.0],
            "yaw": 0.1,
        },
    )
    # The local capture sequence is newer, but the Unitree wall timestamp
    # moved backwards.  Pose must still advance instead of freezing at 0.1.
    state.update_frame(
        frame_seq=11,
        frame_stamp=1001.0,
        telemetry={
            "received_at": 999.0,
            "position": [0.2, 0.0, 0.0],
            "yaw": 1.2,
        },
    )
    telemetry = state.snapshot()["telemetry"]
    assert telemetry["position"] == [0.2, 0.0, 0.0]
    assert telemetry["yaw"] == 1.2


def test_odom_echo_cannot_overwrite_capture_heading():
    """The web arrow must keep the RGB-D pose when the odom mirror echoes it.

    ``/physical_nav/odom`` is generated from the same capture, but arrives on
    a separate ROS callback and has no bridge sequence.  An identity/stale
    quaternion on that lane used to overwrite the fresh scalar Go2 yaw and
    make panels 2/4 look fixed while the source telemetry was changing.
    """
    state = RuntimeState()
    state.update_frame(
        frame_seq=1,
        frame_stamp=100.0,
        telemetry={
            "received_at": 99.95,
            "position": [0.0, 0.0, 0.0],
            "yaw": 0.4,
        },
    )
    state.update_topic(
        "telemetry",
        {
            "received_at": 100.0,
            "position": [0.0, 0.0, 0.0],
            "yaw": 0.0,
            "pose_source": "physical_nav_odom",
        },
    )
    assert state.snapshot()["telemetry"]["yaw"] == 0.4
    # A genuinely newer odom sample remains usable as a fallback/update.
    state.update_topic(
        "telemetry",
        {
            "received_at": 100.2,
            "position": [0.1, 0.0, 0.0],
            "yaw": 0.6,
            "pose_source": "physical_nav_odom",
        },
    )
    assert state.snapshot()["telemetry"]["yaw"] == 0.6


def test_transport_generation_accepts_pose_after_sensor_bridge_restart():
    """A persistent web gateway must accept a reset bridge sequence/clock."""
    state = RuntimeState()
    state.update_frame(
        frame_seq=100,
        frame_stamp=1000.0,
        telemetry_transport_session="sensor-old",
        telemetry_transport_connection=1,
        telemetry={
            "received_at": 1000.0,
            "position": [0.0, 0.0, 0.0],
            "yaw": 0.3,
        },
    )
    state.update_frame(
        frame_seq=1,
        frame_stamp=10.0,
        telemetry_transport_session="sensor-new",
        telemetry_transport_connection=1,
        telemetry={
            "received_at": 10.0,
            "position": [0.2, 0.0, 0.0],
            "yaw": 1.4,
        },
    )
    telemetry = state.snapshot()["telemetry"]
    assert telemetry["position"] == [0.2, 0.0, 0.0]
    assert telemetry["yaw"] == 1.4


def test_telemetry_only_mirror_preserves_transport_generation():
    """A telemetry-only HTTP/ROS receipt must reset pose freshness too."""
    from physical_sensor_ros_bridge import _attach_transport_metadata

    old = RuntimeState()
    old.update_topic(
        "telemetry",
        _attach_transport_metadata(
            {"received_at": 1000.0, "position": [0.0, 0.0, 0.0], "yaw": 0.2},
            "sensor-old",
            1,
        ),
    )
    # Simulate the first telemetry-only mirror after the bridge restarts.  Its
    # source clock and sequence are both lower than the previous process.
    new_payload = _attach_transport_metadata(
        {"received_at": 10.0, "position": [0.3, 0.0, 0.0], "yaw": 1.3},
        "sensor-new",
        1,
    )
    old.update_topic("telemetry", new_payload)
    telemetry = old.snapshot()["telemetry"]
    assert telemetry["position"] == [0.3, 0.0, 0.0]
    assert telemetry["yaw"] == 1.3
    assert "_transport_session" not in telemetry


def test_late_old_transport_generation_cannot_rewind_heading():
    state = RuntimeState()
    state.update_frame(
        frame_seq=5,
        frame_stamp=50.0,
        telemetry_transport_session="sensor-old",
        telemetry_transport_connection=1,
        telemetry={"received_at": 50.0, "yaw": 0.5},
    )
    state.update_frame(
        frame_seq=1,
        frame_stamp=10.0,
        telemetry_transport_session="sensor-new",
        telemetry_transport_connection=1,
        telemetry={"received_at": 10.0, "yaw": 1.5},
    )
    state.update_topic(
        "telemetry",
        {
            "_transport_session": "sensor-old",
            "_transport_connection": 1,
            "received_at": 51.0,
            "yaw": -0.8,
        },
    )
    assert state.snapshot()["telemetry"]["yaw"] == 1.5


def test_legacy_unmarked_telemetry_cannot_overwrite_tagged_pose():
    """A delayed legacy socket may merge diagnostics, never a stale pose."""
    state = RuntimeState()
    state.update_frame(
        frame_seq=4,
        frame_stamp=4.0,
        telemetry_transport_session="sensor-live",
        telemetry_transport_connection=1,
        telemetry={
            "received_at": 4.0,
            "position": [1.0, 2.0, 0.0],
            "yaw": 0.4,
            "battery": {"soc": 90},
        },
    )
    # Legacy telemetry has no generation markers.  It can carry a newer
    # battery sample, but its pose must not rewind the tagged stream.
    state.update_topic(
        "telemetry",
        {
            "received_at": 5.0,
            "position": [9.0, 9.0, 0.0],
            "yaw": -2.0,
            "battery": {"soc": 89},
        },
    )
    telemetry = state.snapshot()["telemetry"]
    assert telemetry["position"] == [1.0, 2.0, 0.0]
    assert telemetry["yaw"] == 0.4
    assert telemetry["battery"]["soc"] == 89


def test_direct_pose_lane_holds_off_odom_race_for_half_second():
    """The capture-time telemetry lane wins over a sequence-less odom echo."""
    from physical_ros_gateway import PhysicalRosGateway

    assert PhysicalRosGateway._direct_pose_is_recent(10.0, 10.49, 0.5)
    assert PhysicalRosGateway._direct_pose_is_recent(10.0, 10.50, 0.5)
    assert not PhysicalRosGateway._direct_pose_is_recent(10.0, 10.51, 0.5)
    assert not PhysicalRosGateway._direct_pose_is_recent(0.0, 10.0, 0.5)
    # A malformed backwards monotonic sample must not suppress the fallback.
    assert not PhysicalRosGateway._direct_pose_is_recent(10.0, 9.9, 0.5)


def test_grid_origin_heading_is_subtracted_in_raster_axes():
    origin = {"qz": math.sin(math.pi / 4), "qw": math.cos(math.pi / 4)}
    assert math.isclose(_grid_origin_yaw(origin), math.pi / 2, abs_tol=1e-6)
    step = {"pose": [1.0, 2.0, 1.2]}
    grid = type("Grid", (), {"origin_yaw": math.pi / 2})()
    adjusted = SixPanelRenderer._step_for_grid_axes(step, grid)
    assert adjusted["pose"][:2] == [1.0, 2.0]
    assert math.isclose(adjusted["pose"][2], 1.2 - math.pi / 2, abs_tol=1e-6)
    assert step["pose"][2] == 1.2


def test_raw_grid_uses_full_origin_quaternion_yaw():
    # A tilted origin has non-zero qx/qy terms; the planar qz-only shortcut
    # gives a different heading and makes the dashboard arrow drift.
    from physical_six_panel_server import SixPanelRenderer

    origin = {"qx": 0.2, "qy": -0.1, "qz": 0.3, "qw": 0.9}
    payload = {"width": 1, "height": 1, "resolution": 0.1,
               "origin": origin, "data": [0]}
    grid = SixPanelRenderer._raw_grid(payload)
    assert grid is not None
    assert math.isclose(grid.origin_yaw, _grid_origin_yaw(origin), abs_tol=1e-12)


def test_persistent_renderer_releases_locked_occ_view_for_new_map_session():
    renderer = SixPanelRenderer(RuntimeState())
    renderer._occ_view_bounds = (-2.0, -2.0, 2.0, 2.0)

    # The renderer may have received a legacy, token-less grid before the
    # first new-process token.  The first token must still release that stale
    # viewport so a restarted map can centre on its initial pose.
    assert renderer._observe_occ_map_session({"map_session": "ros-epoch-a"}) is True
    assert renderer._occ_view_bounds is None
    renderer._occ_view_bounds = (-2.0, -2.0, 2.0, 2.0)
    assert renderer._observe_occ_map_session({"map_session": "ros-epoch-b"}) is True
    assert renderer._occ_view_bounds is None


def test_occ_panel_fits_newly_explored_cells_without_startup_crop(monkeypatch):
    import numpy as np

    state = RuntimeState()
    renderer = SixPanelRenderer(state)
    captured = []
    original = renderer._canonical.render_map_panel

    def capture(*args, **kwargs):
        if kwargs.get("title") == "OCC":
            captured.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(renderer._canonical, "render_map_panel", capture)
    def occupancy(end):
        cells = np.full((80, 80), -1, dtype=np.int8)
        cells[12:20, 12:end] = 0
        return {"width": 80, "height": 80, "resolution": 1.0,
                "frame_id": "tf_frame_map", "origin": {"x": -40, "y": -40},
                "data": cells.ravel().tolist()}

    state.occupancy = occupancy(20)
    renderer.render()
    first = captured[-1]["world_bounds"]
    state.occupancy = occupancy(72)
    state.map_revision += 1
    renderer.render()
    current = captured[-1]
    assert current["world_bounds"][2] > first[2]
    assert current["world_bounds"][0] <= -28
    assert current["world_bounds"][2] >= 32
    assert current["view_scale"] == 1.0
    assert current["expand_crop_for_trajectory"] is False


def test_ros_grid_payload_carries_map_session_token():
    from types import SimpleNamespace

    from physical_ros_gateway import PhysicalRosGateway

    origin = SimpleNamespace(
        position=SimpleNamespace(x=-1.0, y=-2.0, z=0.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    msg = SimpleNamespace(
        info=SimpleNamespace(width=2, height=1, resolution=0.1, origin=origin),
        header=SimpleNamespace(frame_id="tf_frame_map"),
        data=[-1, 0],
    )
    payload = PhysicalRosGateway._grid_payload(msg, map_session="ros-epoch-a")
    assert payload["map_session"] == "ros-epoch-a"
    assert payload["origin"]["x"] == -1.0


def test_showcase_raw_occ_uses_origin_rotation_for_world_points():
    from showcase_pages import ACADEMIC_SHOWCASE_HTML, DARK_SHOWCASE_HTML

    # Both presentation templates must use the shared inverse-origin rotation
    # instead of subtracting only origin.x/y.  The strict academic template is
    # generated from the same source file and is checked as well.
    assert "gridWorldPixel" in DARK_SHOWCASE_HTML
    assert "academicGridPixel" in ACADEMIC_SHOWCASE_HTML
    assert "drawNavigationOverlay" not in DARK_SHOWCASE_HTML
    assert "lx=Math.cos(yaw)*dx+Math.sin(yaw)*dy" in ACADEMIC_SHOWCASE_HTML


def test_sensor_ros_capture_sequence_advances_tf_pose_after_clock_rollback():
    from physical_sensor_ros_bridge import SensorRosBridge

    bridge = object.__new__(SensorRosBridge)
    bridge._telemetry_lock = threading.Lock()
    bridge._telemetry_history = deque(maxlen=64)
    bridge._telemetry_source_stamp = float("-inf")
    bridge._telemetry_capture_seq = -1
    bridge.telemetry = {}
    bridge.args = type("Args", (), {"telemetry_max_delta_sec": 0.15})()
    bridge.telemetry_pub = type("Publisher", (), {"publish": lambda *_args, **_kwargs: None})()
    bridge._mirror_enabled = False
    bridge.update_telemetry(
        {"received_at": 1000.0, "position": [0.0, 0.0, 0.0], "yaw": 0.1},
        1000.0,
        capture_seq=10,
    )
    bridge.update_telemetry(
        {"received_at": 999.0, "position": [0.2, 0.0, 0.0], "yaw": 1.2},
        1001.0,
        capture_seq=11,
    )
    assert bridge.telemetry["yaw"] == 1.2
    assert bridge.telemetry_at(1001.0)[0]["position"] == [0.2, 0.0, 0.0]


def test_sensor_ros_telemetry_mirror_carries_transport_generation():
    """A persistent web gateway must order mirror-only pose packets too."""
    from physical_sensor_ros_bridge import SensorRosBridge

    bridge = object.__new__(SensorRosBridge)
    bridge._telemetry_lock = threading.Lock()
    bridge._telemetry_history = deque(maxlen=64)
    bridge._telemetry_source_stamp = float("-inf")
    bridge._telemetry_capture_seq = -1
    bridge._telemetry_transport_session = None
    bridge._telemetry_transport_connection = -1
    bridge._telemetry_retired_sessions = set()
    bridge.telemetry = {}
    bridge.args = type("Args", (), {"telemetry_max_delta_sec": 0.15})()
    bridge.telemetry_pub = type("Publisher", (), {"publish": lambda *_args, **_kwargs: None})()
    bridge._mirror_enabled = True
    bridge._mirror_lock = threading.Lock()
    bridge._latest_mirror_telemetry = None
    bridge._mirror_event = threading.Event()
    bridge.update_telemetry(
        {"received_at": 10.0, "position": [0.0, 0.0, 0.0], "yaw": 1.0},
        10.0,
        transport_session="sensor-new",
        transport_connection=3,
    )
    mirrored = bridge._latest_mirror_telemetry
    assert mirrored["_transport_session"] == "sensor-new"
    assert mirrored["_transport_connection"] == 3


def test_sensor_queue_uses_transport_seq_when_source_clock_rolls_back():
    """A newer source frame is retained even when its wall stamp is lower."""
    from physical_sensor_ros_bridge import SensorRosBridge

    bridge = object.__new__(SensorRosBridge)
    bridge.last_stamp = float("-inf")
    bridge.last_seq = -1
    bridge._packet_lock = threading.Lock()
    bridge._packet_event = threading.Event()
    bridge._latest_packet = None
    bridge._mirror_enabled = False
    bridge._transport_session = "local"
    bridge._transport_connection = 0
    bridge._active_transport_session = None
    bridge._active_transport_connection = -1
    bridge._retired_transport_sessions = set()
    bridge._bridge_seq = 0
    bridge._source_seq = -1

    packet = lambda seq, stamp, session="go2", connection=1: {
        "seq": seq,
        "stamp": stamp,
        "_transport_session": session,
        "_transport_connection": connection,
    }
    bridge.enqueue_sensor_packet(packet(10, 1000.0))
    bridge.enqueue_sensor_packet(packet(11, 999.0))
    assert bridge._latest_packet["seq"] == 11
    assert bridge._latest_packet["_bridge_seq"] == 2
    bridge.enqueue_sensor_packet(packet(1, 998.0, session="new", connection=1))
    # A delayed packet from the retired source cannot replace the new stream.
    bridge.enqueue_sensor_packet(packet(9, 998.0, session="go2", connection=1))
    assert bridge._latest_packet["seq"] == 1


def test_yolo_claim_accepts_bridge_seq_after_large_clock_rollback():
    from physical_yoloe_bridge import YoloeWorker

    worker = object.__new__(YoloeWorker)
    worker.last_seq = 500
    worker.last_stamp = 1000.0
    frame = {"seq": 1, "stamp": 10.0, "rgb": "rgb", "depth": "depth"}
    assert worker._claim_frame(frame) is True
    assert worker.last_seq == 1
