import math
from collections import deque
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


PACKAGE_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SCRIPTS))

from semantic_mapping_py_pkg.room_segmentation import RoomSegmenter
from semantic_mapping_py_pkg.semantic_occ_overlay import SemanticOccupancyOverlay


def _grid(width=24, height=16, resolution=0.25, yaw=0.0):
    values = np.full((height, width), 100, dtype=np.int8)
    values[1:-1, 1:-1] = 0
    wall_x = width // 2
    values[1:-1, wall_x] = 100
    values[6:10, wall_x] = 0
    origin = SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0),
        orientation=SimpleNamespace(
            x=0.0,
            y=0.0,
            z=math.sin(yaw * 0.5),
            w=math.cos(yaw * 0.5),
        ),
    )
    info = SimpleNamespace(width=width, height=height, resolution=resolution, origin=origin)
    return SimpleNamespace(info=info, data=values.reshape(-1).tolist())


def _door_observation(
    instance_id="door_1",
    center_x=3.0,
    center_y=1.875,
    size_xy=(0.15, 1.0),
):
    return {
        "id": instance_id,
        "name": "door",
        "bbox_2d": [0, 0, 9, 9],
        "segmentation": {
            "rows": [index // 10 for index in range(100)],
            "cols": [index % 10 for index in range(100)],
        },
        "box_3d": {
            "center": [center_x, center_y, 1.0],
            "size": [size_xy[0], size_xy[1], 2.0],
            "frame_id": "world",
        },
    }


def _room_count(room_ids):
    return len({int(room_id) for room_id in room_ids if int(room_id) >= 0})


def _segmenter(**kwargs):
    config = {
        "room_min_component_cells": 4,
        "room_core_min_component_cells": 4,
        "room_core_clearance_cells": 1,
        "room_remove_enclosed_occupied": False,
        "room_portal_cut_margin_m": 0.0,
        "room_portal_cut_thickness_cells": 1,
    }
    config.update(kwargs)
    return RoomSegmenter(**config)


def test_realtime_gt_portal_hint_splits_connected_occupancy_immediately():
    grid = _grid()
    segmenter = _segmenter()

    room_ids_before, _ = segmenter.segment(grid)
    assert _room_count(room_ids_before) == 1

    assert segmenter.update_portal_hints(
        [_door_observation()],
        source_mode="realtime_gt_observation",
    )
    room_ids_after, _ = segmenter.segment(grid)
    assert _room_count(room_ids_after) == 2


def test_portal_cut_transforms_world_endpoints_into_rotated_grid_frame():
    grid = _grid(yaw=math.pi / 2.0)
    segmenter = _segmenter()

    room_ids_before, _ = segmenter.segment(grid)
    assert _room_count(room_ids_before) == 1

    # The original doorway spans local-grid y.  With a +90 degree map origin,
    # that span lies on the world x axis, so the portal AABB major axis changes.
    local_x, local_y = 3.0, 1.875
    center_x = -local_y
    center_y = local_x
    assert segmenter.update_portal_hints(
        [
            _door_observation(
                center_x=center_x,
                center_y=center_y,
                size_xy=(1.0, 0.15),
            )
        ],
        source_mode="realtime_gt_observation",
    )
    room_ids_after, _ = segmenter.segment(grid)

    assert _room_count(room_ids_after) == 2


def test_room_grid_signature_changes_when_only_origin_yaw_changes():
    assert RoomSegmenter._grid_signature(_grid(yaw=0.0).info) != RoomSegmenter._grid_signature(
        _grid(yaw=math.pi / 2.0).info
    )


def test_room_cores_do_not_connect_across_a_diagonal_free_space_pinhole():
    cv2 = pytest.importorskip("cv2")
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[0, 0] = 1
    mask[1, 1] = 1

    count, labels, _stats, _centroids = (
        RoomSegmenter._connected_components_with_stats(mask, cv2)
    )

    assert count == 3
    assert labels[0, 0] != labels[1, 1]


def test_enclosed_obstacle_filter_removes_compact_furniture_but_keeps_wall_like_island():
    width, height = 16, 12
    values = np.zeros((height, width), dtype=np.int8)
    values[0, :] = values[-1, :] = 100
    values[:, 0] = values[:, -1] = 100
    values[3:5, 3:5] = 100  # compact furniture-like island
    values[3:5, 8:13] = 100  # detached but wall-like 2x5 divider
    grid = _grid(width=width, height=height)
    grid.data = values.reshape(-1).tolist()
    segmenter = RoomSegmenter(
        room_min_component_cells=1,
        room_core_min_component_cells=1,
        room_core_clearance_cells=1,
        room_remove_enclosed_occupied=True,
        room_enclosed_occupied_max_cells=100,
        room_enclosed_occupied_max_aspect=1.8,
        room_enclosed_occupied_known_ring_ratio=0.95,
        room_enclosed_occupied_free_ring_ratio=0.8,
        room_portal_cut_enabled=False,
    )

    room_ids, _room_conf = segmenter.segment(grid, force_stable=True)

    assert room_ids[3 * width + 3] >= 0
    assert room_ids[3 * width + 9] == segmenter.room_unknown_id


def _small_portal_pocket_grid():
    width, height = 20, 12
    values = np.full((height, width), 100, dtype=np.int8)
    values[1:11, 1:12] = 0
    values[4:8, 13:17] = 0
    values[5, 12] = 0
    origin = SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    info = SimpleNamespace(width=width, height=height, resolution=1.0, origin=origin)
    return SimpleNamespace(info=info, data=values.reshape(-1).tolist())


def _legacy_free_components_with_labels(segmentation_free):
    """Reference for the pre-P0 Python four-connected flood-fill."""

    height, width = segmentation_free.shape
    flat = segmentation_free.reshape(height * width)
    visited = np.zeros(height * width, dtype=bool)
    labels = np.full(height * width, -1, dtype=np.int32)
    components = []
    for index in range(height * width):
        if visited[index] or flat[index] <= 0:
            continue
        visited[index] = True
        queue = deque([index])
        component = [index]
        component_id = len(components)
        labels[index] = component_id
        while queue:
            current = queue.popleft()
            x = current % width
            y = current // width
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = x + dx
                ny = y + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                next_index = ny * width + nx
                if visited[next_index] or flat[next_index] <= 0:
                    continue
                visited[next_index] = True
                labels[next_index] = component_id
                component.append(next_index)
                queue.append(next_index)
        components.append(component)
    return components, labels


def _legacy_portal_separated_small_components(
    segmenter,
    segmentation_free,
    pre_portal_cut_free,
    portal_cut_mask,
    core_component_cells,
):
    """Reference selection semantics before native four-labeling replaced BFS."""

    height, width = segmentation_free.shape
    post_components, _ = _legacy_free_components_with_labels(segmentation_free)
    if len(post_components) < 2:
        return []
    _pre_components, pre_labels = _legacy_free_components_with_labels(
        pre_portal_cut_free
    )
    seeded_cells = {
        int(cell)
        for component in core_component_cells.values()
        for cell in component
    }
    components_by_pre_cut = {}
    for component in post_components:
        pre_cut_component_id = int(pre_labels[component[0]])
        if pre_cut_component_id < 0:
            continue
        components_by_pre_cut.setdefault(pre_cut_component_id, []).append(component)

    preserved = []
    for components in components_by_pre_cut.values():
        if len(components) < 2:
            continue
        for component in components:
            if len(component) < segmenter.room_min_component_cells:
                continue
            if any(cell in seeded_cells for cell in component):
                continue
            if not segmenter._component_touches_mask(
                component,
                portal_cut_mask,
                width,
                height,
            ):
                continue
            preserved.append(component)
    return preserved


def _portal_cut_fixture():
    """Two sides of one cut plus an unrelated free component."""

    pre = np.zeros((12, 20), dtype=np.uint8)
    pre[1:11, 1:19] = 1
    # The upper-right island was never connected to the cut room.
    pre[0, 0] = 1

    portal_cut = np.zeros_like(pre)
    portal_cut[1:11, 10] = 1
    post = pre.copy()
    post[portal_cut > 0] = 0
    seeded_left_room_cell = 4 * pre.shape[1] + 4
    return pre, post, portal_cut, {1: [seeded_left_room_cell]}


def test_portal_pocket_native_four_labels_match_legacy_selection():
    cv2 = pytest.importorskip("cv2")
    segmenter = _segmenter(room_min_component_cells=12)
    pre, post, portal_cut, core_component_cells = _portal_cut_fixture()

    expected = _legacy_portal_separated_small_components(
        segmenter,
        post,
        pre,
        portal_cut,
        core_component_cells,
    )
    actual = segmenter._portal_separated_small_components(
        post,
        pre,
        portal_cut,
        core_component_cells,
        post.shape[1],
        post.shape[0],
        cv2=cv2,
    )

    # Cell visitation order is deliberately not an implementation contract;
    # retained components and their scan order are.
    assert [set(component) for component in actual] == [
        set(component) for component in expected
    ]


def test_portal_pocket_native_labels_are_strictly_four_connected():
    cv2 = pytest.importorskip("cv2")
    segmenter = _segmenter(room_min_component_cells=1)
    pre = np.ones((3, 3), dtype=np.uint8)
    portal_cut = np.zeros_like(pre)
    portal_cut[0, 1] = 1
    portal_cut[1, :] = 1
    portal_cut[2, 1] = 1
    post = pre.copy()
    post[portal_cut > 0] = 0

    actual = segmenter._portal_separated_small_components(
        post,
        pre,
        portal_cut,
        {},
        post.shape[1],
        post.shape[0],
        cv2=cv2,
    )

    # Four diagonal cells are one component under 8-connectivity but four
    # distinct components under the portal pocket's required 4-connectivity.
    assert [set(component) for component in actual] == [{0}, {2}, {6}, {8}]


def test_portal_pocket_native_labels_do_not_call_python_flood_fill(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    segmenter = _segmenter(room_min_component_cells=12)
    pre, post, portal_cut, core_component_cells = _portal_cut_fixture()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("portal pocket path must not use Python BFS")

    monkeypatch.setattr(
        RoomSegmenter,
        "_free_components_with_labels",
        staticmethod(fail_if_called),
    )
    result = segmenter._portal_separated_small_components(
        post,
        pre,
        portal_cut,
        core_component_cells,
        post.shape[1],
        post.shape[0],
        cv2=cv2,
    )

    assert len(result) == 1


def test_portal_separated_small_free_space_becomes_low_confidence_room():
    grid = _small_portal_pocket_grid()
    segmenter = RoomSegmenter(
        room_min_component_cells=12,
        room_core_min_component_cells=20,
        room_core_clearance_cells=1,
        room_remove_enclosed_occupied=False,
        room_portal_cut_margin_m=0.0,
        room_portal_cut_thickness_cells=1,
        room_portal_small_component_confidence=70,
    )

    room_ids_before, _ = segmenter.segment(grid)
    assert _room_count(room_ids_before) == 1

    assert segmenter.update_portal_hints(
        [_door_observation(center_x=12.0, center_y=5.0)],
        source_mode="realtime_gt_observation",
    )
    room_ids_after, room_conf_after = segmenter.segment(grid)

    pocket_indices = [
        y * grid.info.width + x
        for y in range(4, 8)
        for x in range(13, 17)
    ]
    pocket_room_ids = {room_ids_after[index] for index in pocket_indices}
    assert _room_count(room_ids_after) == 2
    assert len(pocket_room_ids) == 1
    assert next(iter(pocket_room_ids)) >= 0
    assert {room_conf_after[index] for index in pocket_indices} == {70}


def test_confirmed_portal_overlay_segments_narrow_far_side_with_stable_reference():
    """A successful open must make the observed far side a real room.

    The raw map deliberately keeps the door cell occupied.  The topology path
    receives only the confirmed semantic overlay and uses the original,
    closed-door AABB as its virtual room boundary.
    """

    grid = _small_portal_pocket_grid()
    values = np.asarray(grid.data, dtype=np.int8).reshape(
        grid.info.height, grid.info.width
    )
    values[5, 12] = 100
    grid.data = values.reshape(-1).tolist()
    segmenter = RoomSegmenter(
        room_min_component_cells=12,
        room_core_min_component_cells=20,
        room_core_clearance_cells=1,
        room_remove_enclosed_occupied=False,
        room_portal_cut_margin_m=0.0,
        room_portal_cut_thickness_cells=1,
        room_portal_small_component_confidence=70,
        room_grid_stability_frames=3,
    )
    door = _door_observation(center_x=12.0, center_y=5.0)
    assert segmenter.update_portal_hints(
        [door], source_mode="realtime_gt_observation"
    )
    raw_room_ids, _ = segmenter.segment(grid)
    assert _room_count(raw_room_ids) == 1

    overlay = SemanticOccupancyOverlay(clear_padding_m=0.0)
    portal = {
        "id": "door_1",
        "type": "portal",
        "aabb_center": [12.0, 5.0, 1.0],
        "aabb_size": [0.15, 1.0, 2.0],
        "interaction": {"state": "closed"},
    }
    overlay.update_graph({"nodes": [portal]})
    portal["interaction"] = {"state": "open"}
    overlay.update_graph({"nodes": [portal]})
    planning_data, _mask, stats = overlay.apply(
        grid.info,
        grid.data,
        include_pending=False,
    )
    assert stats["active_portal_ids"] == ["door_1"]

    planning_grid = SimpleNamespace(info=grid.info, data=planning_data)
    room_ids, room_conf = segmenter.segment(planning_grid, force_stable=True)
    pocket_indices = [
        y * grid.info.width + x
        for y in range(4, 8)
        for x in range(13, 17)
    ]
    assert _room_count(room_ids) == 2
    assert {room_ids[index] for index in pocket_indices} == {2}
    assert {room_conf[index] for index in pocket_indices} == {70}


def test_post_open_reference_refresh_replaces_an_active_portal_anchor():
    segmenter = _segmenter()
    assert segmenter.update_portal_hints(
        [_door_observation(center_x=3.0)],
        source_mode="realtime_gt_observation",
    )
    assert segmenter.update_portal_hints(
        [_door_observation(center_x=4.0, size_xy=(1.0, 0.15))],
        source_mode="realtime_gt_observation",
        refresh_active=True,
    )
    hint = segmenter.state.portal_hints["door_1"]
    assert hint["center"] == [4.0, 1.875, 1.0]
    assert hint["size"] == [1.0, 0.15, 2.0]


def test_detector_portal_hint_requires_stable_confirmations_and_freezes_anchor():
    grid = _grid()
    segmenter = _segmenter(
        room_portal_detector_min_confirmations=3,
        room_portal_detector_max_center_jump_m=0.3,
    )

    for center_x in (3.0, 3.05):
        assert not segmenter.update_portal_hints(
            [_door_observation(center_x=center_x)],
            source_mode="detector_online",
        )
    room_ids_before, _ = segmenter.segment(grid)
    assert _room_count(room_ids_before) == 1

    assert segmenter.update_portal_hints(
        [_door_observation(center_x=2.95)],
        source_mode="detector_online",
    )
    frozen_center = list(segmenter.state.portal_hints["door_1"]["center"])
    assert not segmenter.update_portal_hints(
        [_door_observation(center_x=4.0)],
        source_mode="detector_online",
    )
    assert segmenter.state.portal_hints["door_1"]["center"] == frozen_center

    room_ids_after, _ = segmenter.segment(grid)
    assert _room_count(room_ids_after) == 2


def test_room_merge_requires_stable_confirmations_before_report() -> None:
    grid = _grid()
    grid.data = [0] * (grid.info.width * grid.info.height)
    segmenter = _segmenter(
        room_merge_confirmations=2,
        room_id_overlap_ratio=0.20,
    )

    split = np.asarray(grid.data, dtype=np.int8).reshape(grid.info.height, grid.info.width)
    split[:, grid.info.width // 2] = 100
    grid.data = split.reshape(-1).tolist()
    room_ids, _ = segmenter.segment(grid)
    assert _room_count(room_ids) == 2

    split[:, grid.info.width // 2] = 0
    grid.data = split.reshape(-1).tolist()
    segmenter.segment(grid)
    assert segmenter.consume_confirmed_merges() == {}

    segmenter.segment(grid)
    merges = segmenter.consume_confirmed_merges()
    assert len(merges) == 1
    secondary, primary = next(iter(merges.items()))
    assert secondary != primary


@pytest.mark.parametrize("existing_pocket", [False, True], ids=["h9", "h1"])
def test_pending_room_split_survives_occupancy_growth(existing_pocket):
    """H1/H9 froze a mature map when an unconfirmed split kept changing ID."""
    pytest.importorskip("cv2")
    grid = _grid(width=160, height=60, resolution=0.1)
    values = np.full((60, 160), -1, dtype=np.int8)
    values[5:55, 5:120] = 0
    if existing_pocket:
        values[10:40, 135:155] = 0
    grid.data = values.reshape(-1).tolist()
    segmenter = RoomSegmenter(
        room_grid_stability_frames=3,
        room_remove_enclosed_occupied=False,
    )
    original, _ = segmenter.segment(grid)
    old_room = original[30 * 160 + 40]
    assert _room_count(original) == 1 + int(existing_pocket)

    values[5:55, 85] = 100
    values[27:33, 85] = 0
    candidate_room = None
    for frame in range(3):
        values[5:55, 120:121 + frame] = 0
        grid.data = values.reshape(-1).tolist()
        published, _ = segmenter.segment(grid)
        candidate = segmenter.state.candidate_room_ids
        if candidate_room is None:
            candidate_room = candidate[30 * 160 + 100]
        assert candidate_room >= 0 and candidate_room != old_room
        assert candidate[30 * 160 + 100] == candidate_room
        assert segmenter.state.candidate_room_count == frame + 1
        assert segmenter.consume_confirmed_merges() == {}
        if frame < 2:
            assert published == original

    assert _room_count(published) == 2 + int(existing_pocket)
    assert published[30 * 160 + 40] == old_room
    assert published[30 * 160 + 122] == candidate_room
    if existing_pocket:
        assert published[25 * 160 + 145] == original[25 * 160 + 145]
    settled_next_id = segmenter.state.next_room_segment_id
    assert segmenter.segment(grid)[0] == published
    assert segmenter.state.next_room_segment_id == settled_next_id


def test_pending_portal_split_keeps_candidate_id_until_grid_confirmation():
    grid = _grid()
    segmenter = _segmenter(room_grid_stability_frames=3)
    original, _ = segmenter.segment(grid)
    segmenter.update_portal_hints(
        [_door_observation()], source_mode="realtime_gt_observation"
    )

    first, _ = segmenter.segment(grid)
    candidate = list(segmenter.state.candidate_room_ids)
    assert first == original
    assert _room_count(candidate) == 2
    assert segmenter.segment(grid)[0] == original
    assert segmenter.segment(grid)[0] == candidate
    assert segmenter.consume_confirmed_merges() == {}


def test_small_split_still_requires_all_stability_frames():
    grid = _grid(width=120, height=80, resolution=1.0)
    values = np.full((80, 120), 100, dtype=np.int8)
    values[1:70, 1:100] = 0
    values[30:34, 102:106] = 0
    values[31, 100:102] = 0
    grid.data = values.reshape(-1).tolist()
    segmenter = _segmenter(room_grid_stability_frames=3)
    original, _ = segmenter.segment(grid)
    assert _room_count(original) == 1
    segmenter.update_portal_hints(
        [_door_observation(center_x=100.0, center_y=31.0)],
        source_mode="realtime_gt_observation",
    )
    for frame in range(3):
        published, _ = segmenter.segment(grid)
        assert segmenter.state.candidate_room_count == frame + 1
        if frame < 2:
            assert published == original
    assert _room_count(published) == 2
    old = np.asarray(original)
    new = np.asarray(published)
    common = (old >= 0) & (new >= 0)
    assert np.mean(old[common] == new[common]) > 0.97


def test_pending_split_keeps_ids_when_component_size_order_changes():
    grid = _grid(width=48, height=16)
    values = np.full((16, 48), -1, dtype=np.int8)
    values[1:15, 1:38] = 0
    grid.data = values.reshape(-1).tolist()
    segmenter = _segmenter(room_grid_stability_frames=3)
    original, _ = segmenter.segment(grid)
    values[:, 20] = 100
    grid.data = values.reshape(-1).tolist()
    assert segmenter.segment(grid)[0] == original
    left_id = segmenter.state.candidate_room_ids[8 * 48 + 10]
    right_id = segmenter.state.candidate_room_ids[8 * 48 + 30]
    assert left_id != right_id

    values[1:15, 38:44] = 0
    grid.data = values.reshape(-1).tolist()
    assert segmenter.segment(grid)[0] == original
    published, _ = segmenter.segment(grid)
    assert published[8 * 48 + 10] == left_id
    assert published[8 * 48 + 30] == right_id
    assert published.count(right_id) > published.count(left_id)
    assert segmenter.consume_confirmed_merges() == {}


@pytest.mark.parametrize("grid_frames,merge_frames", [(3, 2), (2, 4)])
def test_merge_notification_waits_for_stable_grid_commit(grid_frames, merge_frames):
    grid = _grid()
    split = np.asarray(grid.data, dtype=np.int8).reshape(
        grid.info.height, grid.info.width
    )
    split[:, grid.info.width // 2] = 100
    grid.data = split.reshape(-1).tolist()
    segmenter = _segmenter(
        room_grid_stability_frames=grid_frames,
        room_merge_confirmations=merge_frames,
    )
    original, _ = segmenter.segment(grid)
    original_ids = {value for value in original if value >= 0}
    assert len(original_ids) == 2

    split[6:10, grid.info.width // 2] = 0
    grid.data = split.reshape(-1).tolist()
    for _ in range(max(grid_frames, merge_frames) - 1):
        assert segmenter.segment(grid)[0] == original
        assert segmenter.consume_confirmed_merges() == {}
    published, _ = segmenter.segment(grid)
    assert _room_count(published) == 1
    primary = next(value for value in published if value >= 0)
    assert segmenter.consume_confirmed_merges() == {
        secondary: primary for secondary in original_ids if secondary != primary
    }
    for _ in range(3):
        assert segmenter.segment(grid)[0] == published
        assert segmenter.consume_confirmed_merges() == {}


def test_transient_merge_restores_both_stable_room_ids_without_redirects():
    grid = _grid()
    values = np.asarray(grid.data, dtype=np.int8).reshape(
        grid.info.height, grid.info.width
    )
    values[:, grid.info.width // 2] = 100
    grid.data = values.reshape(-1).tolist()
    segmenter = _segmenter(room_grid_stability_frames=3, room_merge_confirmations=2)
    original, _ = segmenter.segment(grid)
    assert _room_count(original) == 2

    values[6:10, grid.info.width // 2] = 0
    grid.data = values.reshape(-1).tolist()
    for _ in range(2):
        assert segmenter.segment(grid)[0] == original
        assert segmenter.consume_confirmed_merges() == {}

    values[:, grid.info.width // 2] = 100
    grid.data = values.reshape(-1).tolist()
    for _ in range(3):
        assert segmenter.segment(grid)[0] == original
        assert segmenter.consume_confirmed_merges() == {}


def test_transient_split_does_not_redirect_the_surviving_stable_room():
    grid = _grid()
    segmenter = _segmenter(room_grid_stability_frames=3)
    original, _ = segmenter.segment(grid)
    values = np.asarray(grid.data, dtype=np.int8).reshape(
        grid.info.height, grid.info.width
    )
    values[:, grid.info.width // 2] = 100
    grid.data = values.reshape(-1).tolist()
    assert segmenter.segment(grid)[0] == original
    values[6:10, grid.info.width // 2] = 0
    grid.data = values.reshape(-1).tolist()
    for _ in range(3):
        assert segmenter.segment(grid)[0] == original
        assert segmenter.consume_confirmed_merges() == {}
