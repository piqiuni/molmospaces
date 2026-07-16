from __future__ import annotations

import numpy as np

from scripts.InteractiveNav import explore_molmo_interactions as emi
from scripts.InteractiveNav import benchmark_door_state_scan as door_scan


def portal_record() -> dict[str, object]:
    return {
        "name": "doorway_test",
        "portal_center_xy": [0.0, 0.0],
        "portal_tangent_xy": [0.0, 1.0],
        "portal_normal_xy": [1.0, 0.0],
        "portal_half_width_m": 0.5,
        "portal_half_thickness_m": 0.05,
        "aabb_center": [0.0, 0.0, 1.0],
        "aabb_size": [0.1, 1.0, 2.0],
    }


def test_triangle_horizontal_slice_preserves_segment() -> None:
    emi.ensure_runtime_dependencies()
    triangle = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
        ]
    )

    segment = emi._triangle_horizontal_slice_segment(triangle, 0.5)

    assert segment is not None
    np.testing.assert_allclose(
        np.asarray(sorted(segment.tolist())),
        np.asarray([[0.0, 1.0], [1.0, 0.0]]),
    )


def test_triangle_horizontal_slice_rejects_non_intersection() -> None:
    emi.ensure_runtime_dependencies()
    triangle = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.1],
            [0.0, 2.0, 0.2],
        ]
    )

    assert emi._triangle_horizontal_slice_segment(triangle, 0.5) is None


def test_rasterize_world_segments_keeps_subpixel_wall() -> None:
    emi.ensure_runtime_dependencies()
    segments = np.asarray([[[1.0, 2.0], [3.0, 2.0]]], dtype=float)
    world_to_map = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    mask = emi.rasterize_world_xy_segments(
        segments,
        world_to_map,
        (5, 5),
        thickness_px=1,
    )

    assert mask.dtype == np.bool_
    assert mask[1, 2]
    assert mask[2, 2]
    assert mask[3, 2]


def test_oriented_xy_bounds_handles_rotated_door_frame() -> None:
    emi.ensure_runtime_dependencies()
    model = emi.mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <compiler angle="degree"/>
          <worldbody>
            <body pos="1.2 -0.4 0.0" euler="0 0 30">
              <geom name="frame_visual" type="box" size="1.0 0.05 0.05"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = emi.mujoco.MjData(model)
    emi.mujoco.mj_forward(model, data)

    bounds = emi.oriented_xy_bounds_for_geoms(model, data, [0])

    assert bounds is not None
    np.testing.assert_allclose(bounds["center_xy"], [1.2, -0.4], atol=1e-6)
    np.testing.assert_allclose(
        bounds["tangent_xy"],
        [np.cos(np.pi / 6.0), np.sin(np.pi / 6.0)],
        atol=1e-6,
    )
    np.testing.assert_allclose(bounds["width_m"], 2.0, atol=1e-6)
    np.testing.assert_allclose(bounds["thickness_m"], 0.1, atol=1e-6)


def test_portal_crossing_requires_opposite_sides() -> None:
    path = np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=float)

    details = door_scan.path_door_crossing_details(
        path,
        portal_record(),
        padding_m=0.1,
        sample_step_m=0.05,
    )

    assert details["traverses"]
    np.testing.assert_allclose(details["crossing_xy"], [0.0, 0.0], atol=1e-6)


def test_portal_tangent_graze_is_not_crossing() -> None:
    path = np.asarray([[0.3, -1.0], [0.3, 1.0]], dtype=float)

    assert not door_scan.path_traverses_door_region(
        path,
        portal_record(),
        padding_m=0.1,
        sample_step_m=0.05,
    )


def test_start_inside_portal_only_ignores_initial_door_exit() -> None:
    initial_exit = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    later_recrossing = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=float
    )

    exit_details = door_scan.path_door_crossing_details(
        initial_exit,
        portal_record(),
        padding_m=0.1,
        sample_step_m=0.05,
    )
    recrossing_details = door_scan.path_door_crossing_details(
        later_recrossing,
        portal_record(),
        padding_m=0.1,
        sample_step_m=0.05,
    )

    assert exit_details["start_inside"]
    assert exit_details["ignored_initial_region"]
    assert not exit_details["traverses"]
    assert recrossing_details["start_inside"]
    assert recrossing_details["ignored_initial_region"]
    assert recrossing_details["traverses"]


def test_start_inside_one_portal_does_not_hide_later_door_crossing() -> None:
    path = np.asarray([[0.0, 0.0], [3.0, 0.0]], dtype=float)
    initial_door = portal_record()
    later_door = portal_record()
    later_door["name"] = "doorway_later"
    later_door["portal_center_xy"] = [2.0, 0.0]

    initial_details = door_scan.path_door_crossing_details(
        path,
        initial_door,
        padding_m=0.1,
        sample_step_m=0.05,
    )
    later_details = door_scan.path_door_crossing_details(
        path,
        later_door,
        padding_m=0.1,
        sample_step_m=0.05,
    )

    assert initial_details["ignored_initial_region"]
    assert not initial_details["traverses"]
    assert not later_details["start_inside"]
    assert later_details["traverses"]
