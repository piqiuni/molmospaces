"""Offline checks for the D435i camera-IMU TF correction helpers.

These tests intentionally do not start ROS or a RealSense pipeline.  They
cover the safety contract used by ``physical_sensor_ros_bridge``: calibration
and invalid samples leave the nominal extrinsic untouched, yaw is opt-in, and
the correction rotates the camera translation as well as its orientation.
"""

import math
import pathlib
import sys
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physical_sensor_ros_bridge import (  # noqa: E402
    _apply_camera_imu_correction,
    _camera_imu_correction_quaternion,
    _camera_imu_parent_correction_quaternion,
    _quat_multiply,
    _quat_rpy,
    _quat_rotate,
)
from physical_yoloe_bridge import (  # noqa: E402
    _OPTICAL_TO_BASE,
    _quaternion_matrix,
    _rotation,
    _telemetry_quaternion,
    _world_transform,
)


class CameraImuTfTests(unittest.TestCase):
    def test_missing_or_calibrating_sample_is_identity(self):
        identity = (0.0, 0.0, 0.0, 1.0)
        self.assertEqual(_camera_imu_correction_quaternion({}), (identity, False))
        self.assertEqual(
            _camera_imu_correction_quaternion(
                {"camera_imu": {"imu_calibrating": True, "correction_rpy": [0.2, 0.1, 0.0]}}
            ),
            (identity, False),
        )

    def test_roll_pitch_are_used_but_relative_yaw_is_ignored_by_default(self):
        telemetry = {
            "camera_imu": {
                # A large integrated yaw must not disable gravity correction
                # when yaw is intentionally disabled.
                "imu_calibrating": False,
                "correction_rpy": [0.1, -0.2, 4.0],
            }
        }
        quaternion, valid = _camera_imu_correction_quaternion(telemetry)
        self.assertTrue(valid)
        self.assertTrue(
            all(
                math.isclose(a, b, abs_tol=1e-7)
                for a, b in zip(quaternion, _quat_rpy(0.1, -0.2, 0.0))
            )
        )
        yaw_quaternion, yaw_valid = _camera_imu_correction_quaternion(
            telemetry, use_yaw=True, max_yaw_rad=5.0
        )
        self.assertTrue(yaw_valid)
        self.assertGreater(abs(yaw_quaternion[2]), abs(quaternion[2]))

    def test_correction_rotates_mount_translation(self):
        correction = _quat_rpy(0.1, 0.0, 0.0)
        _, corrected_translation = _apply_camera_imu_correction(
            (0.0, 0.0, 0.0, 1.0), (0.03, 0.0, 0.98), correction
        )
        expected = _quat_rotate(correction, (0.03, 0.0, 0.98))
        self.assertEqual(corrected_translation, expected)
        self.assertNotAlmostEqual(corrected_translation[1], 0.0, places=5)
        self.assertAlmostEqual(corrected_translation[2], 0.98 * math.cos(0.1), places=5)

    def test_sensor_axes_are_conjugated_into_parent_axes(self):
        telemetry = {
            "camera_imu": {
                "imu_calibrating": False,
                "correction_rpy": [0.1, 0.0, 0.0],
            }
        }
        optical_mount = (0.5, -0.5, 0.5, -0.5)
        parent_quaternion, valid = _camera_imu_parent_correction_quaternion(
            telemetry, optical_mount
        )
        self.assertTrue(valid)
        # D435 +x (right) maps to base -y for this optical mounting.  A
        # sensor-frame roll therefore rotates about base y, not base x.
        rotated_base_z = _quat_rotate(parent_quaternion, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(rotated_base_z[0], -math.sin(0.1), places=5)
        self.assertAlmostEqual(rotated_base_z[1], 0.0, places=5)

    def test_yolo_embedded_world_transform_matches_corrected_sensor_tf(self):
        """Fallback boxes must use the same corrected rigid transform as TF.

        This exercises both the D435 depth->colour baseline and the horizontal
        translation induced when the elevated camera mount leans.  A mismatch
        here is otherwise easy to miss because the normal gateway path
        transforms camera segments with TF and only uses embedded world points
        while TF is temporarily unavailable.
        """
        rpy = (0.0, 0.1396263, 0.0)
        translation = (0.03, -0.02, 0.98)
        depth_to_color = {
            "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "translation": [0.025, 0.001, 0.0],
        }
        yaw = 0.31
        telemetry = {
            "position": [1.2, -0.4, 0.27],
            # Unitree's body quaternion is wxyz in the wire telemetry.
            "imu": {
                "quaternion": [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
            },
            "camera_imu": {
                "imu_calibrating": False,
                "correction_rpy": [0.08, -0.06, 0.0],
            },
        }
        rotation, offset = _world_transform(
            telemetry,
            np.asarray(translation, dtype="float32"),
            rpy,
            optical_frame=True,
            depth_to_color_extrinsics=depth_to_color,
        )

        mount_quaternion = _quat_multiply(
            _quat_rpy(*rpy), (0.5, -0.5, 0.5, -0.5)
        )
        correction_quaternion, valid = _camera_imu_parent_correction_quaternion(
            telemetry, mount_quaternion
        )
        self.assertTrue(valid)
        correction_rotation = _quaternion_matrix(correction_quaternion)
        body_rotation = _quaternion_matrix(
            (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
        )
        extrinsic_rotation = np.asarray(
            depth_to_color["rotation"], dtype="float32"
        ).reshape(3, 3)
        extrinsic_translation = np.asarray(
            depth_to_color["translation"], dtype="float32"
        )
        mount_rotation = np.asarray(_rotation(*rpy), dtype="float32")
        nominal_rotation = mount_rotation @ _OPTICAL_TO_BASE @ extrinsic_rotation
        nominal_translation = (
            mount_rotation @ _OPTICAL_TO_BASE @ extrinsic_translation
            + np.asarray(translation, dtype="float32")
        )
        expected_rotation = body_rotation @ correction_rotation @ nominal_rotation
        expected_offset = (
            body_rotation @ (correction_rotation @ nominal_translation)
            + np.asarray(telemetry["position"], dtype="float32")
        )
        self.assertTrue(
            np.allclose(rotation, expected_rotation, atol=2e-6)
        )
        self.assertTrue(
            np.allclose(offset, expected_offset, atol=2e-6)
        )

    def test_yolo_uses_same_body_imu_as_sensor_odom_when_pose_fields_coexist(self):
        """A camera-pose hint must not override the odom body quaternion."""
        body_yaw = 0.23
        body = (math.cos(body_yaw / 2.0), 0.0, 0.0, math.sin(body_yaw / 2.0))
        camera_hint = (0.0, 0.0, math.sin(1.1 / 2.0), math.cos(1.1 / 2.0))
        value = _telemetry_quaternion(
            {
                "imu": {"quaternion": list(body)},
                "camera_pose": {"quaternion": list(camera_hint)},
            }
        )
        self.assertIsNotNone(value)
        self.assertTrue(
            np.allclose(
                value,
                (body[1], body[2], body[3], body[0]),
                atol=1e-7,
            )
        )

    def test_yolo_can_follow_sensor_correction_disable_gate(self):
        telemetry = {
            "position": [0.0, 0.0, 0.0],
            "imu": {"quaternion": [1.0, 0.0, 0.0, 0.0]},
            "camera_imu": {
                "imu_calibrating": False,
                "correction_rpy": [0.2, -0.1, 0.0],
            },
        }
        kwargs = {
            "telemetry": telemetry,
            "translation": np.asarray([0.03, 0.0, 0.98], dtype="float32"),
            "rpy": (0.0, 0.1396263, 0.0),
            "optical_frame": True,
        }
        corrected_rotation, corrected_offset = _world_transform(**kwargs)
        nominal_rotation, nominal_offset = _world_transform(
            **kwargs, camera_imu_enabled=False
        )
        self.assertGreater(
            float(np.max(np.abs(corrected_rotation - nominal_rotation))), 1e-3
        )
        self.assertGreater(
            float(np.linalg.norm(corrected_offset - nominal_offset)), 1e-3
        )


if __name__ == "__main__":
    unittest.main()
