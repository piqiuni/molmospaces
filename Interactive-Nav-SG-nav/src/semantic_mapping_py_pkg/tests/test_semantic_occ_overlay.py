from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from semantic_mapping_py_pkg.semantic_occ_overlay import OverlayUpdateRegionTracker, SemanticOccupancyOverlay


class GridInfo:
    width = 20
    height = 20
    resolution = 0.1

    class Origin:
        class Position:
            x = 0.0
            y = 0.0

        class Orientation:
            x = 0.0
            y = 0.0
            z = 0.0
            w = 1.0

        position = Position()
        orientation = Orientation()

    origin = Origin()


def portal(state, center=(1.0, 1.0, 1.0), size=(0.8, 0.1, 2.0)):
    return {
        "id": "portal_door",
        "type": "portal",
        "aabb_center": list(center),
        "aabb_size": list(size),
        "interaction": {"state": state},
    }


def graph(node):
    return {"nodes": [node]}


def test_closed_portal_keeps_raw_map_and_open_portal_clears_persistently():
    overlay = SemanticOccupancyOverlay(clear_padding_m=0.0)
    raw = [100] * (GridInfo.width * GridInfo.height)

    overlay.update_graph(graph(portal("closed")))
    closed_data, closed_mask, closed_stats = overlay.apply(GridInfo(), raw)
    assert closed_data == raw
    assert max(closed_mask) == 0
    assert closed_stats["cleared_cells"] == 0

    overlay.update_graph(graph(portal("open", center=(1.5, 1.5, 1.0))))
    open_data, open_mask, open_stats = overlay.apply(GridInfo(), raw)
    assert open_stats["active_portal_ids"] == ["portal_door"]
    assert open_stats["cleared_cells"] > 0
    assert max(open_mask) == 100
    # The cached closed AABB is cleared; the moved open-state AABB is not used.
    assert open_data[10 * GridInfo.width + 10] == 0
    assert open_data[15 * GridInfo.width + 15] == 100

    # A later raw map can reintroduce occupied values, but the semantic overlay
    # must clear the active portal again.
    repeated_data, _mask, repeated_stats = overlay.apply(GridInfo(), [100] * len(raw))
    assert repeated_stats["cleared_cells"] == open_stats["cleared_cells"]
    assert repeated_data[10 * GridInfo.width + 10] == 0


def test_closing_portal_removes_active_overlay():
    overlay = SemanticOccupancyOverlay(clear_padding_m=0.0)
    raw = [100] * (GridInfo.width * GridInfo.height)
    overlay.update_graph(graph(portal("closed")))
    overlay.update_graph(graph(portal("open")))
    assert overlay.apply(GridInfo(), raw)[2]["cleared_cells"] > 0

    overlay.update_graph(graph(portal("closed")))
    restored, mask, stats = overlay.apply(GridInfo(), raw)
    assert restored == raw
    assert max(mask) == 0
    assert stats["cleared_cells"] == 0


def test_reset_clears_cross_episode_portal_state():
    overlay = SemanticOccupancyOverlay()
    overlay.update_graph(graph(portal("closed")))
    overlay.update_graph(graph(portal("open")))
    assert overlay.active_portal_ids == {"portal_door"}
    overlay.reset()
    assert overlay.active_portal_ids == set()
    assert overlay.reference_aabbs == {}


def test_overlay_reports_the_small_door_update_region():
    overlay = SemanticOccupancyOverlay(clear_padding_m=0.0)
    raw = [100] * (GridInfo.width * GridInfo.height)
    overlay.update_graph(graph(portal("closed")))
    overlay.update_graph(graph(portal("open")))
    _planning, mask, stats = overlay.apply(GridInfo(), raw)

    assert stats["update_bounds"] == {"x": 5, "y": 9, "width": 9, "height": 2}
    assert sum(value > 0 for value in mask) == 18


def test_update_region_is_persistent_and_restores_previous_door_region_on_close():
    tracker = OverlayUpdateRegionTracker(retired_bounds_hold_builds=2)
    width = 6
    height = 5
    geometry = (width, height, 0.1)
    opened = [100] * (width * height)
    for row in range(1, 3):
        for col in range(2, 5):
            opened[row * width + col] = 0

    first = tracker.build(
        width,
        height,
        opened,
        {"x": 2, "y": 1, "width": 3, "height": 2},
        geometry_key=geometry,
    )
    repeated = tracker.build(
        width,
        height,
        opened,
        {"x": 2, "y": 1, "width": 3, "height": 2},
        geometry_key=geometry,
    )
    restored_map = [100] * (width * height)
    restored = tracker.build(width, height, restored_map, None, geometry_key=geometry)
    repeated_restore = tracker.build(width, height, restored_map, None, geometry_key=geometry)
    quiet = tracker.build(width, height, restored_map, None, geometry_key=geometry)

    expected_bounds = {"x": 2, "y": 1, "width": 3, "height": 2}
    assert {key: first[key] for key in expected_bounds} == expected_bounds
    assert first["data"] == [0] * 6
    assert repeated == first
    assert {key: restored[key] for key in expected_bounds} == expected_bounds
    assert restored["data"] == [100] * 6
    assert repeated_restore == restored
    assert quiet is None


def test_update_region_also_forwards_non_overlay_occupancy_changes():
    tracker = OverlayUpdateRegionTracker()
    width = 6
    height = 5
    geometry = (width, height, 0.1)
    initial = [-1] * (width * height)
    assert tracker.build(width, height, initial, None, geometry_key=geometry) is None

    changed = list(initial)
    changed[1 * width + 4] = 0
    changed[3 * width + 2] = 100
    update = tracker.build(width, height, changed, None, geometry_key=geometry)

    assert {key: update[key] for key in ("x", "y", "width", "height")} == {
        "x": 2,
        "y": 1,
        "width": 3,
        "height": 3,
    }
    assert update["data"] == [-1, -1, 0, -1, -1, -1, 100, -1, -1]
