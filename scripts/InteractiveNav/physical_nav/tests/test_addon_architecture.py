"""Deployment boundaries: no ROS master, GPU, web server or robot required."""

import ast
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from physical_nav_watchdog import HealthLimits, PipelineProgress, health_errors
from semantic_decision_py_pkg.approach_geometry import portal_approach_pose
from semantic_decision_py_pkg.progress_clock import ProgressClock
from semantic_mapping_py_pkg.semantic_evidence import persistence_action


ADDON = Path(__file__).resolve().parents[1]
REPO = ADDON.parents[2]
ROS_PACKAGE = REPO / "Interactive-Nav-SG-nav/src/physical_nav"


@pytest.mark.parametrize("asset", [
    "config/physical_nav.yaml", "config/semantic_shadow_override.yaml",
    "config/explore_physical_override.yaml", "config/physical_move_base_override.yaml",
    "launch/physical_nav_readonly.launch",
])
def test_ros_assets_are_aliases_not_independent_copies(asset):
    assert (ROS_PACKAGE / asset).is_symlink()
    assert (ROS_PACKAGE / asset).resolve() == (ADDON / asset).resolve()


@pytest.mark.parametrize("name", ["physical_nav_readonly.launch", "physical_sensor_ros_bridge.py"])
def test_ros_resource_lookup_is_unambiguous(name):
    from roslib.packages import _find_resource
    matches = _find_resource(str(ROS_PACKAGE), name)
    assert len(matches) == 1, matches


def test_runtime_entry_supports_a_relocated_install(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "physical_nav_runtime", ROS_PACKAGE / "scripts/physical_nav_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    (runtime / "physical_protocol.py").write_text("# marker\n")
    (runtime / "adapter.py").write_text("# fake entry\n")
    captured = []
    monkeypatch.setattr(module.runpy, "run_path", lambda path, **kw: captured.append(
        (path, kw["run_name"], sys.path[0])))
    before = list(sys.path)
    module.run("adapter.py", package_root=tmp_path)
    assert captured == [(str(runtime / "adapter.py"), "__main__", str(runtime))]
    assert sys.path == before
    with pytest.raises(ValueError):
        module.run("../adapter.py", package_root=tmp_path)


def test_core_does_not_import_device_or_web_addons():
    forbidden = {p.stem for p in ADDON.glob("*.py")} | {"physical_nav"}
    for package in ("semantic_mapping_py_pkg", "semantic_decision_py_pkg", "semantic_mllm_py_pkg"):
        for file in (REPO / "Interactive-Nav-SG-nav/src" / package / "scripts").rglob("*.py"):
            for node in ast.walk(ast.parse(file.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    names = [(node.module or "").split(".")[0]]
                else:
                    continue
                assert not forbidden.intersection(names), (file, names)


def test_capture_geometry_does_not_import_ros_or_node_implementations():
    import capture_geometry
    import physical_sensor_ros_bridge as sensor
    import physical_yoloe_bridge as yolo
    assert sensor._nearest_capture_telemetry is capture_geometry._nearest_capture_telemetry
    assert yolo._nearest_capture_telemetry is capture_geometry._nearest_capture_telemetry
    tree = ast.parse((ADDON / "capture_geometry.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module in {"__future__", "typing"}
        elif isinstance(node, ast.Import):
            assert all(alias.name in {"math", "numpy"} for alias in node.names)


def test_ros_watchdog_cannot_be_kept_alive_by_duplicates_or_out_of_order_data():
    now = [10.0]
    progress = PipelineProgress(clock=lambda: now[0])
    assert health_errors(progress.snapshot(), HealthLimits())
    for stream in ("frame", "perception"):
        assert progress.observe(stream, 1, 100.0, ros_now=100.1)
    assert not health_errors(progress.snapshot(), HealthLimits())
    now[0] += 20.0
    assert not progress.observe("frame", 99, 100.0)
    assert not progress.observe("perception", 99, 99.0)
    errors = health_errors(progress.snapshot(), HealthLimits())
    assert len(errors) == 2 and all("stale" in error for error in errors)
    # Sensor restart may reset seq; the normalized capture stamp remains increasing.
    for stream in ("frame", "perception"):
        assert progress.observe(stream, 0, 120.0, ros_now=120.1)
    assert not health_errors(progress.snapshot(), HealthLimits())


def test_ros_watchdog_rejects_queued_old_frames_and_bad_values():
    progress = PipelineProgress()
    assert not progress.observe("frame", None, 1.0)
    assert not progress.observe("frame", 1, float("nan"))
    assert progress.observe("frame", 1, 100.0, ros_now=130.0)
    errors = health_errors(progress.snapshot(), HealthLimits(require_perception=False))
    assert len(errors) == 1 and "stale" in errors[0]


def test_headless_supervisor_still_monitors_ros_progress():
    source = (ADDON / "start_physical_nav.sh").read_text()
    section = source[source.index('if [[ "${PHYSICAL_NAV_WATCHDOG_ENABLED'):
                     source.index('log_supervisor "component logs:')]
    condition = section.splitlines()[0]
    assert "START_WEB" not in condition
    assert "--source ros" in section
    assert "/api/health" not in section
    assert "GATEWAY_READY" not in source
    assert "sha256sum \"${ROOT_DIR}/capture_geometry.py\"" in source


@pytest.mark.parametrize("publish_fails", [False, True])
def test_yolo_heartbeat_follows_report_success_not_worker_loop(monkeypatch, publish_fails):
    import physical_yoloe_bridge as module
    worker = object.__new__(module.YoloeWorker)
    worker._report_lock = threading.Lock()
    worker._report_event = threading.Event()
    worker._report_event.set()
    worker._pending_report = {"seq": 4, "stamp": 100.5, "detections": []}
    events = []

    def report(message):
        events.append("report")
        if publish_fails:
            raise RuntimeError("test transport failure")

    worker.report_pub = SimpleNamespace(publish=report)
    worker.heartbeat_pub = SimpleNamespace(publish=lambda message: events.append(message))
    shutdown = iter([False, True])
    monkeypatch.setattr(module.rospy, "is_shutdown", lambda: next(shutdown))
    monkeypatch.setattr(module.rospy, "logwarn_throttle", lambda *args: None)
    worker._report_publish_loop()
    assert events[0] == "report"
    if publish_fails:
        assert len(events) == 1
    else:
        assert events[1].seq == 4 and events[1].stamp.to_sec() == 100.5


def test_clock_policy_preserves_capture_identity():
    assert ProgressClock().context(4, now=100)["observation_step"] == 4
    result = ProgressClock("monotonic", .2).context(4, now=100)
    assert result["observation_step"] == 500 and result["source_capture_step"] == 4
    for mode, period in (("unknown", .2), ("monotonic", 0), ("monotonic", float("nan"))):
        with pytest.raises(ValueError):
            ProgressClock(mode, period)


def test_approach_policy_has_no_hidden_state_and_returns_side_memory():
    node = {"id": "door", "aabb_center": [0, 0, 1], "aabb_size": [1, .1, 2],
            "attributes": {"interaction_reference_yaw": 0.0}}
    pose, side = portal_approach_pose((0, 1), (0, 0), node, .8)
    assert pose[:2] == pytest.approx([0, .85]) and side == 1.0
    pose, side = portal_approach_pose((0, -.01), (0, 0), node, .8,
                                     previous_side=side, side_hysteresis_m=.1)
    assert pose[1] == pytest.approx(.85) and side == 1.0
    pose, side = portal_approach_pose((0, -1), (0, 0), node, .8,
                                     previous_side=side, side_hysteresis_m=.1)
    assert pose[1] == pytest.approx(-.85) and side == -1.0


def test_semantic_policy_does_not_mutate_graph_or_admit_single_frame():
    attrs = {"attribute_status": "ready", "attribute_confidence": .9,
             "mllm_interaction_class": "container", "m1_observed_object_name": "refrigerator"}
    before = dict(attrs)
    assert persistence_action("container", 1, attrs) == "keep"
    assert persistence_action("container", 2, attrs) == "latch"
    assert persistence_action("portal", 2, attrs) == "clear"
    pending = {"attribute_status": "pending", "attribute_last_ready": attrs}
    assert persistence_action("container", 2, pending) == "latch"
    assert attrs == before
