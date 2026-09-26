from copy import deepcopy
import math

import numpy as np
import pytest

from go2_readonly_sensor_bridge import _upright_rgbd_180


def intrinsics(width, height):
    return dict(width=width, height=height, fx=100., fy=120., cx=width * .4,
                cy=height * .6, distortion_model="distortion.brown_conrady",
                distortion=[.01, .02, .03, .04, .05])


def test_rotation_preserves_depth_samples_native_sizes_and_input_calibration():
    rgb = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(6, 8, 3)
    depth = np.arange(4 * 3, dtype=np.uint16).reshape(3, 4)
    ci, di = intrinsics(8, 6), intrinsics(4, 3)
    angle = .1
    rotation = np.array([[math.cos(angle), 0, math.sin(angle)],
                         [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
    extrinsics = dict(rotation=rotation.ravel().tolist(), translation=[.015, -.002, .001])
    before = deepcopy((ci, di, extrinsics))
    color, distance, cnew, dnew, enew = _upright_rgbd_180(rgb, depth, ci, di, extrinsics)
    np.testing.assert_array_equal(color, rgb[::-1, ::-1])
    np.testing.assert_array_equal(distance, depth[::-1, ::-1])
    assert distance.dtype == np.uint16
    assert distance.flags.c_contiguous and color.flags.c_contiguous
    assert not np.shares_memory(distance, depth)
    assert not np.shares_memory(color, rgb)
    assert (ci, di, extrinsics) == before
    assert cnew["cx"] == pytest.approx(7 - ci["cx"])
    assert dnew["cy"] == pytest.approx(2 - di["cy"])
    assert cnew["distortion"] == [.01, .02, -.03, -.04, .05]
    signs = np.diag([-1., -1., 1.])
    np.testing.assert_allclose(np.array(enew["rotation"]).reshape(3, 3), signs @ rotation @ signs)
    np.testing.assert_allclose(enew["translation"], [-.015, .002, .001])
    # Rotating twice recovers the original calibration and exact depth values.
    c2, d2, ci2, di2, ex2 = _upright_rgbd_180(color, distance, cnew, dnew, enew)
    np.testing.assert_array_equal(c2, rgb)
    np.testing.assert_array_equal(d2, depth)
    assert ci2 == ci and di2 == di and ex2 == extrinsics


@pytest.mark.parametrize("angle", [0., .02, -.03])
def test_inverted_rgb_mask_projection_and_depth_tf_are_consistent(angle):
    ci, di = intrinsics(1280, 720), intrinsics(848, 480)
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0],
                         [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    translation = np.array([.014, -.003, .002])
    _, _, cnew, dnew, enew = _upright_rgbd_180(
        np.zeros((720, 1280, 3), np.uint8), np.zeros((480, 848), np.uint16),
        ci, di, dict(rotation=rotation.ravel().tolist(), translation=translation.tolist()),
    )
    u, v, z = 321., 123., 2.
    native_point = np.array([(u - di["cx"]) / di["fx"] * z, (v - di["cy"]) / di["fy"] * z, z])
    upright_point = np.array([(847 - u - dnew["cx"]) / dnew["fx"] * z,
                              (479 - v - dnew["cy"]) / dnew["fy"] * z, z])
    signs = np.diag([-1., -1., 1.])
    np.testing.assert_allclose(upright_point, signs @ native_point)
    native_color = rotation @ native_point + translation
    upright_color = np.array(enew["rotation"]).reshape(3, 3) @ upright_point + enew["translation"]
    np.testing.assert_allclose(upright_color, signs @ native_color)
    old_uv = native_color[:2] / native_color[2] * [ci["fx"], ci["fy"]] + [ci["cx"], ci["cy"]]
    new_uv = upright_color[:2] / upright_color[2] * [cnew["fx"], cnew["fy"]] + [cnew["cx"], cnew["cy"]]
    np.testing.assert_allclose(new_uv, [1279, 719] - old_uv)
    # Physical native mount has a 180-degree roll; upright virtual mount does not.
    upright_mount = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    np.testing.assert_allclose(upright_mount @ upright_color,
                               upright_mount @ signs @ native_color)
