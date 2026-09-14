import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package in ("explore_py_pkg", "semantic_decision_py_pkg"):
    sys.path.insert(0, str(ROOT / "Interactive-Nav-SG-nav/src" / package / "scripts"))

from explore_py_pkg.room_colors import room_color
from explore_py_pkg.debug_semantic_viz import semantic_node_display_label
from semantic_decision_py_pkg.navigation_arrival import path_deviation_m


def test_room_colors_are_stable_and_do_not_repeat_at_eight():
    assert room_color(2) != room_color(10)
    assert len({room_color(i) for i in range(128)}) == 128
    assert [room_color(i) for i in (10, 2)] == [room_color(10), room_color(2)]


def test_room_labels_use_inferred_name_without_appending_room():
    for value, expected in (
        ("bedroom", "bedroom"),
        ("dining_room", "dining room"),
        ("livingroom", "living room"),
        ("living room", "living room"),
        ("kitchen", "kitchen"),
        ("unknown", "unknown"),
        (None, "unknown"),
        ("  ", "unknown"),
    ):
        node = {"type": "room", "attributes": {"room_attribute": value}}
        assert semantic_node_display_label(node) == expected
    assert semantic_node_display_label({"type": "portal", "label": "door_0001"}) == "door_0001"


def test_path_deviation_uses_segments_and_endpoints():
    assert path_deviation_m((1, 0, 0), [(0, 0), (2, 0)]) == 0
    assert path_deviation_m((1, 0.5, 0), [(0, 0), (2, 0)]) == 0.5
    assert path_deviation_m((3, 0, 0), [(0, 0), (2, 0)]) == 1
    assert path_deviation_m((1, 0, 0), [(0, 0), (0, 0)]) == 1


def test_raw_simulation_entry_enables_physical_odometry_twist():
    tree = ast.parse((ROOT / "scripts/InteractiveNav/run_nav_ros_sim.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "RosBridgePolicy"]
    assert calls
    for call in calls:
        assert any(k.arg == "publish_odom_twist" and isinstance(k.value, ast.Constant)
                   and k.value.value is True for k in call.keywords)
        assert any(k.arg == "odom_twist_source" and isinstance(k.value, ast.Constant)
                   and k.value.value == "step_delta" for k in call.keywords)
