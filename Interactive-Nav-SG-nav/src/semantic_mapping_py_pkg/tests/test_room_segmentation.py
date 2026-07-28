import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


PACKAGE_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SCRIPTS))

from semantic_mapping_py_pkg.room_segmentation import RoomSegmenter


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
    return RoomSegmenter(
        room_min_component_cells=4,
        room_core_min_component_cells=4,
        room_core_clearance_cells=1,
        room_remove_enclosed_occupied=False,
        room_portal_cut_margin_m=0.0,
        room_portal_cut_thickness_cells=1,
        **kwargs,
    )


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
