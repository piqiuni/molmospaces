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
    assert overlay.pending_portal_ids == set()


def test_confirmed_open_survives_graph_gap_and_later_raw_door_reappearance():
    overlay = SemanticOccupancyOverlay(
        clear_padding_m=0.0,
        raw_free_confirmations=3,
    )
    blocked = [100] * (GridInfo.width * GridInfo.height)
    overlay.update_graph(graph(portal("closed")))
    overlay.update_graph(graph(portal("open")))

    # A post-open traversal timeout can temporarily publish a graph without the
    # portal.  That must not restore the stale closed-door cells.
    overlay.update_graph({"nodes": []})
    planning, _mask, stats = overlay.apply(GridInfo(), blocked)
    assert stats["active_portal_ids"] == ["portal_door"]
    assert planning[10 * GridInfo.width + 10] == 0

    raw_free = list(blocked)
    for index, value in enumerate(planning):
        if value == 0:
            raw_free[index] = 0
    for _ in range(2):
        _planning, _mask, stats = overlay.apply(GridInfo(), raw_free)
        assert stats["active_portal_ids"] == ["portal_door"]

    planning, mask, stats = overlay.apply(GridInfo(), raw_free)
    assert stats["active_portal_ids"] == ["portal_door"]
    assert planning == raw_free
    assert max(mask) == 100

    # A later depth frame may project the moved leaf back into its old aperture.
    # Public open state remains authoritative until an explicit close, so the
    # stale occupied cells are still cleared.
    planning, _mask, stats = overlay.apply(GridInfo(), blocked)
    assert stats["active_portal_ids"] == ["portal_door"]
    assert planning[10 * GridInfo.width + 10] == 0

    overlay.update_graph(graph(portal("open")))
    assert overlay.has_active_portals()
    overlay.update_graph(graph(portal("closed")))
    assert not overlay.has_active_portals()
    overlay.update_graph(graph(portal("open")))
    assert overlay.has_active_portals()


def test_pending_open_interaction_clears_before_result_and_rolls_back():
    overlay = SemanticOccupancyOverlay(clear_padding_m=0.0)
    raw = [100] * (GridInfo.width * GridInfo.height)
    overlay.update_graph(graph(portal("closed")))

    assert overlay.set_interaction_pending("portal_door", True) is True
    overlay.update_graph(graph(portal("closed")))
    pending_data, _mask, pending_stats = overlay.apply(GridInfo(), raw)
    assert pending_stats["active_portal_ids"] == ["portal_door"]
    assert pending_data[10 * GridInfo.width + 10] == 0

    # Room topology must not treat an in-flight skill as an observed opening.
    room_data, room_mask, room_stats = overlay.apply(
        GridInfo(), raw, include_pending=False
    )
    assert room_data == raw
    assert max(room_mask) == 0
    assert room_stats["active_portal_ids"] == []

    assert overlay.set_interaction_pending("portal_door", False) is True
    overlay.update_graph(graph(portal("open")))
    confirmed_data, _mask, confirmed_stats = overlay.apply(
        GridInfo(), raw, include_pending=False
    )
    assert confirmed_stats["active_portal_ids"] == ["portal_door"]
    assert confirmed_data[10 * GridInfo.width + 10] == 0

    overlay.update_graph(graph(portal("closed")))
    restored, _mask, restored_stats = overlay.apply(GridInfo(), raw)
    assert restored == raw
    assert restored_stats["active_portal_ids"] == []


def test_container_open_and_pending_command_never_clear_occupancy():
    overlay = SemanticOccupancyOverlay(clear_padding_m=0.0)
    raw = [100] * (GridInfo.width * GridInfo.height)
    fridge = {
        "id": "container_fridge_1",
        "type": "container",
        "aabb_center": [1.0, 1.0, 1.0],
        "aabb_size": [0.8, 0.8, 2.0],
        "interaction": {"state": "open"},
        "attributes": {"topology_type": "container"},
    }

    overlay.update_graph(graph(fridge))
    # A generic open command must never create an optimistic doorway clear.
    assert not overlay.set_interaction_pending("container_fridge_1", True)
    assert not overlay.set_interaction_pending(
        "container_fridge_1", True, node_type="container"
    )
    planning, mask, stats = overlay.apply(GridInfo(), raw)
    assert planning == raw
    assert max(mask) == 0
    assert stats["active_portal_ids"] == []


def test_mllm_portal_label_cannot_promote_source_container_to_overlay_portal():
    overlay = SemanticOccupancyOverlay(clear_padding_m=0.0)
    raw = [100] * (GridInfo.width * GridInfo.height)
    transient_label = {
        "id": "container_fridge_1",
        # This is the historical bad shape: presentation type changed by M1,
        # while source observation provenance still says container.
        "type": "portal",
        "aabb_center": [1.0, 1.0, 1.0],
        "aabb_size": [0.8, 0.8, 2.0],
        "interaction": {"state": "open"},
        "attributes": {"topology_type": "container"},
    }

    overlay.update_graph(graph(transient_label))
    planning, mask, stats = overlay.apply(GridInfo(), raw)
    assert planning == raw
    assert max(mask) == 0
    assert stats["active_portal_ids"] == []


def test_ajar_portal_keeps_semantic_clearance():
    overlay = SemanticOccupancyOverlay(
        clear_padding_m=0.0,
        open_states=["ajar", "open"],
    )
    raw = [100] * (GridInfo.width * GridInfo.height)
    overlay.update_graph(graph(portal("closed")))
    overlay.update_graph(graph(portal("ajar", center=(1.5, 1.5, 1.0))))

    planning, _mask, stats = overlay.apply(GridInfo(), raw)
    assert stats["active_portal_ids"] == ["portal_door"]
    assert planning[10 * GridInfo.width + 10] == 0


def test_overlay_prefers_immutable_doorway_reference_over_open_leaf_aabb():
    overlay = SemanticOccupancyOverlay(clear_padding_m=0.0)
    raw = [100] * (GridInfo.width * GridInfo.height)
    opened = portal("open", center=(1.5, 1.5, 1.0))
    opened["attributes"] = {
        "interaction_reference_aabb_center": [1.0, 1.0, 1.0],
        "interaction_reference_aabb_size": [0.8, 0.1, 2.0],
    }
    overlay.update_graph(graph(opened))

    planning, _mask, stats = overlay.apply(GridInfo(), raw)
    assert stats["active_portal_ids"] == ["portal_door"]
    assert planning[10 * GridInfo.width + 10] == 0
    assert planning[15 * GridInfo.width + 15] == 100


def test_overlay_reports_the_small_door_update_region():
    overlay = SemanticOccupancyOverlay(clear_padding_m=0.0)
    raw = [100] * (GridInfo.width * GridInfo.height)
    overlay.update_graph(graph(portal("closed")))
    overlay.update_graph(graph(portal("open")))
    _planning, mask, stats = overlay.apply(GridInfo(), raw)

    # A cell must be centred in the doorway slab.  The former overlap-based
    # fill also cleared the two cells touching the door's lateral AABB edge.
    assert stats["update_bounds"] == {"x": 6, "y": 9, "width": 8, "height": 2}
    assert sum(value > 0 for value in mask) == 16


def test_open_portal_insets_reference_and_preserves_adjacent_wall_cells():
    overlay = SemanticOccupancyOverlay(clear_padding_m=-0.05)
    raw = [100] * (GridInfo.width * GridInfo.height)
    overlay.update_graph(graph(portal("closed")))
    overlay.update_graph(graph(portal("open")))

    planning, mask, stats = overlay.apply(GridInfo(), raw)

    assert stats["active_portal_ids"] == ["portal_door"]
    # The 80 cm reference opening is inset to 70 cm, so the wall cells just
    # outside its lateral endpoints remain occupied.
    assert planning[10 * GridInfo.width + 5] == 100
    assert planning[10 * GridInfo.width + 14] == 100
    assert planning[10 * GridInfo.width + 10] == 0
    assert sum(value > 0 for value in mask) == 16


def test_non_grid_aligned_thin_portal_clears_both_occupied_rows():
    overlay = SemanticOccupancyOverlay(
        clear_padding_m=-0.05,
        max_aperture_thickness_m=0.25,
    )
    raw = [100] * (GridInfo.width * GridInfo.height)
    opened = portal(
        "closed",
        center=(1.0, 1.067, 1.0),
        size=(0.8, 0.1697, 2.0),
    )
    overlay.update_graph(graph(opened))
    opened["interaction"] = {"state": "open"}
    overlay.update_graph(graph(opened))

    planning, mask, stats = overlay.apply(GridInfo(), raw)

    assert stats["active_portal_ids"] == ["portal_door"]
    cleared_rows = {
        index // GridInfo.width
        for index, value in enumerate(mask)
        if value > 0
    }
    assert cleared_rows == {10, 11}
    for row in cleared_rows:
        assert all(
            planning[row * GridInfo.width + col] == 0
            for col in range(6, 14)
        )
    # The lateral inset still protects cells immediately outside the opening.
    assert planning[10 * GridInfo.width + 5] == 100
    assert planning[10 * GridInfo.width + 14] == 100


def test_wide_portal_reference_is_limited_to_a_narrow_doorway_slab():
    overlay = SemanticOccupancyOverlay(
        clear_padding_m=0.0,
        max_aperture_thickness_m=0.25,
    )
    raw = [100] * (GridInfo.width * GridInfo.height)
    wide_portal = portal("closed", size=(0.8, 0.8, 2.0))
    overlay.update_graph(graph(wide_portal))
    wide_portal["interaction"] = {"state": "open"}
    overlay.update_graph(graph(wide_portal))

    planning, mask, stats = overlay.apply(GridInfo(), raw)

    assert stats["active_portal_ids"] == ["portal_door"]
    assert planning[10 * GridInfo.width + 10] == 0
    # A coarse/square reference must not clear the entire area around a door.
    assert planning[6 * GridInfo.width + 10] == 100
    assert planning[13 * GridInfo.width + 10] == 100
    assert sum(value > 0 for value in mask) < 32


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
