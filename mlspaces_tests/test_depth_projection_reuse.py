import numpy as np
import pytest

from molmo_spaces.policy.learned_policy.organized_depth_scan import OrganizedDepthScanProjector


def mask_reference(mask, axis, window, min_cover):
    support = np.zeros(mask.shape, dtype=np.int16)
    if mask.shape[axis] < window:
        return support.astype(bool)
    windows = np.lib.stride_tricks.sliding_window_view(mask, window, axis=axis)
    good = np.all(windows, axis=-1)
    for offset in range(window):
        slices = [slice(None)] * mask.ndim
        slices[axis] = slice(offset, offset + good.shape[axis])
        support[tuple(slices)] += good
    return support >= max(1, min_cover)


@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (23, 31)])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("window", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("min_cover", [0, 1, 2, 5])
def test_mask_support_matches_window_reference(shape, axis, window, min_cover):
    random = np.random.default_rng(321)
    for mask in (random.random(shape) > .2, np.ones(shape, dtype=bool), np.zeros(shape, dtype=bool)):
        result = OrganizedDepthScanProjector._mask_support(mask, axis=axis, window=window, min_cover=min_cover)
        np.testing.assert_array_equal(result, mask_reference(mask, axis, window, min_cover))


def test_ray_factors_are_exact_read_only_and_reused():
    projector = OrganizedDepthScanProjector()
    intrinsics = (512., 510., 511.5, 287.5)
    columns, rows = projector._ray_factors((576, 1024), intrinsics)
    np.testing.assert_array_equal(columns, -((np.arange(1024).astype(np.float64) - 511.5) / 512.))
    np.testing.assert_array_equal(rows, -((np.arange(576).astype(np.float64) - 287.5) / 510.))
    assert projector._ray_factors((576, 1024), intrinsics)[0] is columns
    assert projector._ray_factors((576, 1024), intrinsics)[1] is rows
    assert not columns.flags.writeable and not rows.flags.writeable


@pytest.mark.parametrize("change", range(6))
def test_calibration_changes_match_fresh_projector(change):
    projector = OrganizedDepthScanProjector()
    base_shape, base_intrinsics = (24, 32), (25., 26., 15.5, 11.5)
    shape, intrinsics = list(base_shape), list(base_intrinsics)
    if change < 2:
        shape[change] += 2
    else:
        intrinsics[change - 2] += 1.25
    transform = np.eye(4)
    transform[2, 3] = 1.0
    for current_shape, current_intrinsics in ((base_shape, base_intrinsics), (shape, intrinsics),
                                               (base_shape, base_intrinsics)):
        depth = np.full(current_shape, 2., dtype=np.float32)
        depth[4:12, 6:15] = np.nan
        result = projector.project(depth, current_intrinsics, transform)
        expected = OrganizedDepthScanProjector().project(depth, current_intrinsics, transform)
        np.testing.assert_array_equal(result.ranges_m, expected.ranges_m)
        np.testing.assert_array_equal(result.intensities, expected.intensities)
        assert result.diagnostics == expected.diagnostics
