from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace


PACKAGE_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SCRIPTS))

from semantic_mapping_py_pkg.geometry_utils import (
    grid_origin_yaw,
    grid_to_world,
    world_to_grid,
)


def _grid_info(yaw=0.0):
    return SimpleNamespace(
        width=12,
        height=10,
        resolution=0.5,
        origin=SimpleNamespace(
            position=SimpleNamespace(x=10.0, y=-2.0),
            orientation=SimpleNamespace(
                x=0.0,
                y=0.0,
                z=math.sin(yaw * 0.5),
                w=math.cos(yaw * 0.5),
            ),
        ),
    )


def test_grid_world_round_trip_respects_rotated_origin():
    info = _grid_info(math.pi / 2.0)

    world_x, world_y = grid_to_world(2, 3, info)

    assert math.isclose(world_x, 8.25)
    assert math.isclose(world_y, -0.75)
    assert world_to_grid(world_x, world_y, info) == (2, 3)
    assert math.isclose(grid_origin_yaw(info), math.pi / 2.0)


def test_world_to_grid_defaults_to_identity_when_orientation_is_absent():
    info = _grid_info()
    del info.origin.orientation

    assert world_to_grid(11.1, -0.6, info) == (2, 2)
