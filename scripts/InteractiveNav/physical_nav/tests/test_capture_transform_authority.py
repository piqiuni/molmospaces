"""YOLO must use the camera transform that the sensor TF branch publishes."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import physical_yoloe_bridge as yolo
import physical_sensor_ros_bridge as sensor
from test_body_pose_consistency import _bridge
from test_yolo_capture_pose import _worker_and_raw


@pytest.mark.parametrize("native_depth", [False, True])
def test_real_infer_uses_published_sensor_transform_not_stale_local_mount(monkeypatch, native_depth):
    bridge, odom, sent = _bridge(monkeypatch)
    bridge._camera_imu_enabled = True
    bridge.args.camera_x, bridge.args.camera_y, bridge.args.camera_z = .03, .02, .98
    bridge.args.camera_roll, bridge.args.camera_pitch, bridge.args.camera_yaw = .05, .14, -.03
    pose = {"position": [2., 3., .305], "yaw": .4,
            "camera_imu": {"imu_calibrating": False, "correction_rpy": [.1, -.2, .04]}}
    depth_frame = "depth" if native_depth else "camera"
    extrinsics = {"rotation": [1., 0., 0., 0., 1., 0., 0., 0., 1.],
                  "translation": [.03, 0., 0.]}
    bridge.publish_pose(pose, sensor.rospy.Time.from_sec(10.), depth_frame, extrinsics)
    camera = next(t for t in sent[-1] if t.child_frame_id == depth_frame)
    q = camera.transform.rotation
    p = camera.transform.translation
    camera_contract = {"parent_frame": camera.header.frame_id, "child_frame": depth_frame,
                       "quaternion_xyzw": [q.x, q.y, q.z, q.w],
                       "translation": [p.x, p.y, p.z]}
    body_q = odom[0].pose.pose.orientation
    body_rotation = yolo._quaternion_matrix([body_q.x, body_q.y, body_q.z, body_q.w])
    expected_rotation = body_rotation @ yolo._quaternion_matrix(camera_contract["quaternion_xyzw"])
    expected_offset = body_rotation @ np.array(camera_contract["translation"]) + pose["position"]
    worker, raw = _worker_and_raw()
    # Simulate a long-lived YOLO worker with old CLI/ROS parameters.
    worker.translation = np.array([9., 8., 7.])
    worker.args.camera_pitch = -.5
    worker.args.camera_imu_enabled = False
    raw.update(telemetry=pose, camera_frame="camera", depth_frame=depth_frame,
               depth_to_base=camera_contract, depth_to_color_extrinsics=extrinsics,
               rgb_intrinsics=raw["depth_intrinsics"])
    seen = []
    def geometry(*args):
        seen.append(args[-1])
        return {"ok": False, "reason": "offline_depth_rejected"}
    monkeypatch.setattr(yolo, "_geometry_lift_candidate", geometry)
    report = worker.infer(raw)
    assert report["capture_pose_status"] == "valid"
    assert len(seen) == 1
    assert np.allclose(seen[0][0], expected_rotation, atol=1e-6)
    assert np.allclose(seen[0][1], expected_offset, atol=1e-6)
    assert seen[0][0].dtype == seen[0][1].dtype == np.float32


def test_context_and_tf_reuse_one_computed_camera_transform(monkeypatch):
    bridge, _, sent = _bridge(monkeypatch)
    bridge._camera_imu_enabled = True
    original = bridge._camera_transforms
    computations = []
    def compute(*args):
        result = original(*args)
        computations.append(result)
        return result
    bridge._camera_transforms = compute
    payloads = []
    bridge.capture_context_pub = SimpleNamespace(publish=lambda msg: payloads.append(json.loads(msg.data)))
    pose = {"position": [1., 2., .3], "yaw": .4}
    stamp = sensor.rospy.Time.from_sec(10.)
    calibration = {"depth_frame": "depth", "rgb_frame": "camera", "depth_scale": .001,
                   "depth_to_color_extrinsics": {}, "telemetry_max_delta_sec": .15}
    transforms = bridge.publish_capture_context(stamp, 1, pose, 0., {}, {}, calibration)
    bridge.args.camera_z = 9.  # Changing config after capture cannot re-pose this frame.
    bridge.publish_pose(pose, stamp, "depth", {}, camera_transforms=transforms)
    assert len(computations) == 1
    assert all(actual is computed for actual, computed in zip(sent[-1], transforms))
    transform = next(t for t in sent[-1] if t.child_frame_id == "depth")
    contract = payloads[0]["depth_to_base"]
    assert contract["translation"] == [transform.transform.translation.x,
                                       transform.transform.translation.y,
                                       transform.transform.translation.z]
    assert contract["translation"][2] == 1.
    q = transform.transform.rotation
    assert contract["quaternion_xyzw"] == [q.x, q.y, q.z, q.w]


@pytest.mark.parametrize("bad_patch", [
    {"parent_frame": "tf_frame_map"}, {"child_frame": "wrong_camera"},
    {"quaternion_xyzw": [0., 0., 0., 0.]},
    {"quaternion_xyzw": [0., float("nan"), 0., 1.]},
    {"translation": [1., 2.]}, {"translation": [1., 2., 1e300]},
])
def test_invalid_authoritative_transform_retains_2d_and_never_uses_cli(monkeypatch, bad_patch):
    worker, raw = _worker_and_raw()
    raw["telemetry"] = {"position": [1., 2., .3], "yaw": 0.}
    raw["depth_to_base"] = {"parent_frame": "tf_frame_base_link", "child_frame": "depth",
                            "quaternion_xyzw": [0., 0., 0., 1.], "translation": [0., 0., 1.],
                            **bad_patch}
    def unexpected(*args, **kwargs):
        pytest.fail("Rejected transform must not invoke geometry or a legacy world transform")
    monkeypatch.setattr(yolo, "_world_transform", unexpected)
    monkeypatch.setattr(yolo, "_geometry_lift_candidate", unexpected)
    report = worker.infer(raw)
    assert "capture_transform" in report["capture_pose_status"]
    assert report["detections"][0]["geometry_skipped"]
    assert report["overlay_jpeg"]


def test_live_context_missing_transform_does_not_silently_use_old_mount(monkeypatch):
    worker, raw = _worker_and_raw()
    raw["telemetry"] = {"position": [1., 2., .3], "yaw": 0.}
    raw["capture_pose_match"] = {"source": "sensor_capture_context", "accepted": True}
    report = worker.infer(raw)
    assert report["capture_pose_status"] == "missing_capture_transform"
    assert report["detections"][0]["geometry_skipped"]
