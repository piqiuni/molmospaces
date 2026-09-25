import numpy as np
import pytest

from molmo_spaces.policy.learned_policy.organized_depth_scan import OrganizedDepthScanProjector


def legacy_continuity(depth, valid, *, axis, window, min_cover, gap_abs_m, gap_rel):
    support = np.zeros(depth.shape, dtype=np.int16)
    if depth.shape[axis] < window:
        return support.astype(bool)
    windows = np.lib.stride_tricks.sliding_window_view(depth, window, axis=axis)
    valid_windows = np.lib.stride_tricks.sliding_window_view(valid, window, axis=axis)
    adjacent = np.abs(np.diff(windows, axis=-1))
    gap = np.maximum(gap_abs_m, gap_rel*np.minimum(windows[..., :-1], windows[..., 1:]))
    good = np.all(valid_windows, axis=-1) & np.all(adjacent <= gap, axis=-1)
    for offset in range(window):
        slices = [slice(None), slice(None)]
        slices[axis] = slice(offset, offset+good.shape[axis])
        support[tuple(slices)] += good
    return support >= max(1, min_cover)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("window", [1, 2, 3, 5, 21])
@pytest.mark.parametrize("cover", [0, 1, 2, 6])
def test_continuity_matches_overlapping_windows(axis, window, cover):
    rng = np.random.default_rng(516)
    for shape in [(1, 1), (7, 19), (32, 48)]:
        depth = rng.choice(np.array([0., .1, 2., 2.08, 2.09, 3., np.nan, np.inf], dtype=np.float32), shape)
        valid = np.isfinite(depth) & (depth >= .1)
        kwargs = dict(axis=axis, window=window, min_cover=cover, gap_abs_m=.08, gap_rel=.04)
        with np.errstate(invalid="ignore"):
            np.testing.assert_array_equal(OrganizedDepthScanProjector._continuity_support(depth, valid, **kwargs),
                                          legacy_continuity(depth, valid, **kwargs))


def test_full_projection_is_exactly_equivalent(monkeypatch):
    depth = np.full((576, 1024), 3., dtype=np.float32)
    depth[120:350, 500:800] = 1.5
    depth[50:150, 40:90] = np.nan
    transform = np.eye(4)
    transform[2, 3] = 1.
    projector = OrganizedDepthScanProjector()
    actual = projector.project(depth, (512., 512., 512., 288.), transform)
    monkeypatch.setattr(projector, "_continuity_support", legacy_continuity)
    expected = projector.project(depth, (512., 512., 512., 288.), transform)
    np.testing.assert_array_equal(actual.ranges_m, expected.ranges_m)
    np.testing.assert_array_equal(actual.intensities, expected.intensities)
    assert actual.diagnostics == expected.diagnostics
