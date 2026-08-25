import numpy as np

from molmo_spaces.policy.learned_policy.organized_depth_scan import (
    OrganizedDepthScanConfig,
    OrganizedDepthScanProjector,
)


def _projector() -> OrganizedDepthScanProjector:
    return OrganizedDepthScanProjector(
        OrganizedDepthScanConfig(
            angle_increment_deg=0.5,
            vertical_window=5,
            vertical_min_cover=2,
            horizontal_window=3,
            horizontal_min_cover=1,
            min_support_neighbors=1,
        )
    )


def _flat_scene() -> np.ndarray:
    return np.full((64, 128), 3.0, dtype=np.float32)


def test_projection_has_fixed_circular_beams_and_nan_unknowns() -> None:
    scan = _projector().project(
        _flat_scene(),
        (80.0, 80.0, 64.0, 32.0),
        np.eye(4, dtype=np.float64),
    )

    assert scan.ranges_m.shape == (720,)
    assert scan.intensities.shape == (720,)
    assert np.isfinite(scan.ranges_m).any()
    assert not np.isinf(scan.ranges_m).any()
    assert scan.diagnostics["beam_count"] == 720
    assert np.isclose(np.rad2deg(scan.angle_increment_rad), 0.5)


def test_single_column_depth_edge_does_not_become_a_near_obstacle() -> None:
    depth = _flat_scene()
    depth[8:28, 64] = 1.5

    scan = _projector().project(
        depth,
        (80.0, 80.0, 64.0, 32.0),
        np.eye(4, dtype=np.float64),
    )

    finite = scan.ranges_m[np.isfinite(scan.ranges_m)]
    assert finite.size > 0
    assert np.all(finite > 2.5)


def test_contiguous_depth_patch_is_preserved() -> None:
    depth = _flat_scene()
    depth[8:28, 60:69] = 1.5

    scan = _projector().project(
        depth,
        (80.0, 80.0, 64.0, 32.0),
        np.eye(4, dtype=np.float64),
    )

    finite = scan.ranges_m[np.isfinite(scan.ranges_m)]
    assert finite.size > 0
    assert np.any(finite < 2.0)
    assert scan.diagnostics["organized_supported_pixels"] > 0


def test_projection_is_deterministic() -> None:
    projector = _projector()
    depth = _flat_scene()
    first = projector.project(depth, (80.0, 80.0, 64.0, 32.0), np.eye(4))
    second = projector.project(depth, (80.0, 80.0, 64.0, 32.0), np.eye(4))

    np.testing.assert_array_equal(first.ranges_m, second.ranges_m)
    assert first.diagnostics == second.diagnostics


def test_supported_far_depth_emits_no_return_free_space_beams() -> None:
    projector = OrganizedDepthScanProjector(
        OrganizedDepthScanConfig(
            angle_increment_deg=0.5,
            range_max_m=4.0,
            no_return_margin_m=0.05,
            vertical_window=5,
            vertical_min_cover=2,
            horizontal_window=3,
            horizontal_min_cover=1,
            min_support_neighbors=1,
        )
    )
    scan = projector.project(
        np.full((64, 128), 12.0, dtype=np.float32),
        (80.0, 80.0, 64.0, 32.0),
        np.eye(4, dtype=np.float64),
    )

    no_return = scan.intensities == 2.0
    assert no_return.any()
    assert np.allclose(scan.ranges_m[no_return], 3.95)
    assert scan.diagnostics["accepted_no_return_beams"] == int(no_return.sum())


def test_contiguous_no_hit_depth_clears_only_to_scan_horizon() -> None:
    projector = OrganizedDepthScanProjector(
        OrganizedDepthScanConfig(
            angle_increment_deg=0.5,
            range_max_m=4.0,
            no_return_margin_m=0.05,
            vertical_window=5,
            vertical_min_cover=2,
            horizontal_window=3,
            horizontal_min_cover=1,
            min_support_neighbors=1,
        )
    )
    scan = projector.project(
        np.full((64, 128), np.nan, dtype=np.float32),
        (80.0, 80.0, 64.0, 32.0),
        np.eye(4, dtype=np.float64),
    )

    no_return = scan.intensities == 2.0
    assert no_return.any()
    assert np.allclose(scan.ranges_m[no_return], 3.95)
    assert np.all(scan.ranges_m[no_return] < scan.range_max_m)
    assert scan.diagnostics["invalid_no_return_pixels"] > 0
    assert scan.diagnostics["nonfinite_depth_pixels"] == 64 * 128
    assert scan.diagnostics["unsupported_no_hit_pixels"] < 64 * 128
    assert scan.diagnostics["accepted_no_return_free_beams"] == int(
        no_return.sum()
    )
    assert scan.diagnostics["accepted_hit_beams"] == 0
    # The 360-degree scan keeps directions outside the camera FOV unknown.
    assert np.isnan(scan.ranges_m).any()
    assert scan.diagnostics["unknown_output_beams"] == int(
        np.isnan(scan.ranges_m).sum()
    )


def test_supported_downward_no_hit_patch_clears_to_horizon() -> None:
    """No-hit free rays do not depend on their 8 m endpoint height."""

    depth = np.zeros((64, 128), dtype=np.float32)
    depth[52:64, 32:96] = np.nan
    base_from_lidar = np.eye(4, dtype=np.float64)
    base_from_lidar[2, 3] = 1.2

    scan = _projector().project(
        depth,
        (80.0, 40.0, 64.0, 32.0),
        base_from_lidar,
    )

    no_return = scan.intensities == 2.0
    assert no_return.any()
    assert np.allclose(scan.ranges_m[no_return], 7.95)
    assert np.isnan(scan.ranges_m).any()


def test_isolated_invalid_depth_pixel_remains_unknown() -> None:
    depth = np.zeros((64, 128), dtype=np.float32)
    depth[32, 64] = np.nan
    scan = _projector().project(
        depth,
        (80.0, 80.0, 64.0, 32.0),
        np.eye(4, dtype=np.float64),
    )

    assert not np.isfinite(scan.ranges_m).any()
    assert scan.diagnostics["invalid_no_return_pixels"] == 0
    assert scan.diagnostics["nonfinite_depth_pixels"] == 1
    assert scan.diagnostics["unsupported_no_hit_pixels"] == 1
    assert scan.diagnostics["accepted_no_return_free_beams"] == 0
    assert scan.diagnostics["accepted_hit_beams"] == 0


def test_diagnostics_separate_hits_invalid_pixels_and_support_rejections() -> None:
    depth = _flat_scene()
    # A coherent obstacle remains an accepted hit.
    depth[8:28, 60:69] = 1.5
    # An isolated no-hit sample must remain unsupported/unknown.
    depth[32, 10] = np.nan
    # Finite but outside the trusted sensor interval is diagnosed separately.
    depth[32, 110] = 50.0

    scan = _projector().project(
        depth,
        (80.0, 80.0, 64.0, 32.0),
        np.eye(4, dtype=np.float64),
    )

    assert scan.diagnostics["accepted_hit_beams"] == int(
        (scan.intensities == 1.0).sum()
    )
    assert scan.diagnostics["accepted_no_return_free_beams"] == int(
        (scan.intensities == 2.0).sum()
    )
    assert scan.diagnostics["nonfinite_depth_pixels"] == 1
    assert scan.diagnostics["out_of_range_depth_pixels"] == 1
    assert scan.diagnostics["unsupported_no_hit_pixels"] == 1
    assert scan.diagnostics["support_rejected_hit_beams"] >= 0
    assert scan.diagnostics["support_rejected_no_return_beams"] >= 0


def test_sparse_near_depth_edge_falls_back_to_supported_far_free_space() -> None:
    projector = OrganizedDepthScanProjector(
        OrganizedDepthScanConfig(
            angle_increment_deg=0.5,
            range_max_m=4.0,
            no_return_margin_m=0.05,
            vertical_window=5,
            vertical_min_cover=2,
            # Deliberately admit a single-column near edge.  It has no angular
            # support, while the surrounding far surface does, so it must not
            # keep the central observed ray unknown.
            horizontal_window=1,
            horizontal_min_cover=1,
            min_support_neighbors=1,
        )
    )
    depth = np.full((64, 128), 12.0, dtype=np.float32)
    depth[8:28, 64] = 1.5

    scan = projector.project(
        depth,
        (80.0, 80.0, 64.0, 32.0),
        np.eye(4, dtype=np.float64),
    )

    center_beam = scan.ranges_m.shape[0] // 2
    assert scan.intensities[center_beam] == 2.0
    assert np.isclose(scan.ranges_m[center_beam], 3.95)
    assert scan.diagnostics["fallback_no_return_beams"] >= 1


def test_coherent_near_surface_precedes_far_free_space_in_same_bearing() -> None:
    projector = OrganizedDepthScanProjector(
        OrganizedDepthScanConfig(
            angle_increment_deg=0.5,
            range_max_m=4.0,
            no_return_margin_m=0.05,
            vertical_window=5,
            vertical_min_cover=2,
            horizontal_window=3,
            horizontal_min_cover=1,
            min_support_neighbors=1,
        )
    )
    depth = np.full((64, 128), 12.0, dtype=np.float32)
    depth[8:28, 63:66] = 1.5

    scan = projector.project(
        depth,
        (80.0, 80.0, 64.0, 32.0),
        np.eye(4, dtype=np.float64),
    )

    center_beam = scan.ranges_m.shape[0] // 2
    assert scan.intensities[center_beam] == 1.0
    assert scan.ranges_m[center_beam] < 2.0
