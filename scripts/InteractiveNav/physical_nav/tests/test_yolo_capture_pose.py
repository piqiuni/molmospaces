"""No fabricated world detections when a capture has no usable body pose."""

from pathlib import Path
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import physical_yoloe_bridge as yolo


def _worker_and_raw():
    worker = object.__new__(yolo.YoloeWorker)
    worker.args = SimpleNamespace(device="cpu", imgsz=8, conf=.1, iou=.7, max_det=10,
                                  camera_roll=0., camera_pitch=0., camera_yaw=0.,
                                  point_stride=1, max_depth_m=8., min_valid_points=3,
                                  model_path="offline-stub")
    worker.translation = np.array([0., 0., 1.], dtype=np.float32)
    worker.detector_config = {"geometry_budget_ms": 0.}
    worker.class_confidence_thresholds = {}
    worker.detection_filter = SimpleNamespace(apply_one=lambda item: item)
    result = SimpleNamespace(
        boxes=SimpleNamespace(xyxy=np.array([[0., 0., 7., 7.]]),
                              conf=np.array([.95]), cls=np.array([0])),
        masks=SimpleNamespace(data=np.ones((1, 8, 8), dtype=np.float32)),
        names={0: "chair"},
    )
    worker.model = SimpleNamespace(predict=lambda **kwargs: [result])
    raw = {"seq": 1, "stamp": 10., "camera_frame": "depth", "depth_frame": "depth",
           "rgb_array": np.zeros((8, 8, 3), dtype=np.uint8),
           "depth_array": np.ones((8, 8), dtype=np.uint16) * 1000,
           "depth_intrinsics": {"fx": 8., "fy": 8., "cx": 4., "cy": 4.},
           "depth_scale": .001, "telemetry": {}}
    return worker, raw


@pytest.mark.parametrize("pose", [
    {}, None, {"yaw": 0.}, {"position": [1., 2., 3.]},
    {"position": [1., 2.], "yaw": 0.},
    {"position": [1., 2., float("nan")], "yaw": 0.},
    {"position": [1., 2., 3.], "yaw": float("inf")},
    {"position": np.array(1.), "yaw": 0.},
    {"position": [[1.], [2.], [3.]], "yaw": 0.},
])
def test_infer_retains_2d_without_world_geometry_or_projection_work(monkeypatch, pose):
    worker, raw = _worker_and_raw()
    raw["telemetry"] = pose

    def unexpected(*args, **kwargs):
        pytest.fail("No world geometry work is allowed without a capture pose")

    monkeypatch.setattr(yolo, "_geometry_lift_candidate", unexpected)
    monkeypatch.setattr(yolo, "_world_transform", unexpected)
    report = worker.infer(raw)
    assert report["capture_pose_status"] != "valid"
    assert report["seq"] == 1 and report["stamp"] == 10.
    assert report["overlay_jpeg"]
    assert len(report["detections"]) == 1
    detection = report["detections"][0]
    assert detection["semantic_class"] == "chair"
    assert detection["confidence"] == pytest.approx(.95)
    assert detection["geometry_skipped"]
    assert detection["geometry_skip_reason"] == report["capture_pose_status"]
    assert not any(key in detection for key in (
        "position", "world_position", "world_box3d_center", "world_segment_points_f32"))


@pytest.mark.parametrize("pose", [
    {"position": [0., 0., 0.], "yaw": 0.},
    {"position": [5., 6., .3], "imu": {"quaternion": [1., 0., 0., 0.]}},
    {"position": [5., 6., .3], "imu": {"rpy": [0., 0., .5]}},
])
def test_explicit_valid_origin_or_body_orientation_is_accepted(pose):
    assert yolo._capture_pose_error(pose) is None


def test_pose_recovery_reenables_lifting_on_next_frame(monkeypatch):
    worker, raw = _worker_and_raw()
    calls = []

    def geometry(*args):
        calls.append(args[-1])
        return {"ok": False, "reason": "offline_depth_rejected"}

    monkeypatch.setattr(yolo, "_geometry_lift_candidate", geometry)
    missing = worker.infer(raw)
    assert calls == []
    raw.update(seq=2, stamp=10.1, telemetry={"position": [5., 6., .3], "yaw": 0.})
    recovered = worker.infer(raw)
    assert recovered["capture_pose_status"] == "valid"
    assert len(calls) == 1
    assert np.allclose(calls[0][1], [5., 6., 1.3])
    assert missing["detections"][0]["geometry_skip_reason"] == "missing_capture_pose"
    assert recovered["detections"][0]["geometry_skip_reason"] == "offline_depth_rejected"


def test_geometry_deadline_never_blocks_next_gpu_inference(monkeypatch):
    """A slow 3-D lift is abandoned without queueing the following frame."""
    worker, raw = _worker_and_raw()
    raw["telemetry"] = {"position": [0.0, 0.0, 0.0], "yaw": 0.0}
    worker.detector_config["geometry_budget_ms"] = 25.0
    worker._geometry_workers = 1
    worker._geometry_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-yolo-geometry"
    )
    worker._geometry_inflight = set()
    worker._geometry_abandoned_total = 0
    geometry_started = threading.Event()
    release_geometry = threading.Event()
    predict_receipts = []

    original_predict = worker.model.predict

    def predict(**kwargs):
        predict_receipts.append(time.perf_counter())
        return original_predict(**kwargs)

    def slow_geometry(*_args):
        geometry_started.set()
        assert release_geometry.wait(timeout=2.0)
        return {
            "ok": False,
            "reason": "released_slow_geometry",
            "mask_projection_ms": 0.0,
            "depth_geometry_ms": 0.0,
        }

    worker.model.predict = predict
    monkeypatch.setattr(yolo, "_geometry_lift_candidate", slow_geometry)
    try:
        first_started = time.perf_counter()
        first = worker.infer(raw)
        first_elapsed = time.perf_counter() - first_started
        assert geometry_started.is_set()
        assert first_elapsed < 0.5
        assert first["detections"][0]["geometry_skip_reason"] == "geometry_budget_ms"
        first_parts = first["timings_ms"]["post_parts"]
        assert first_parts["geometry_timed_out"] == 1.0
        assert first_parts["geometry_inflight"] == 1.0

        # The only worker is still occupied by frame 1.  Frame 2 must still
        # reach model.predict immediately and must not queue stale geometry.
        raw.update(seq=2, stamp=10.1)
        second_started = time.perf_counter()
        second = worker.infer(raw)
        second_elapsed = time.perf_counter() - second_started
        assert second_elapsed < 0.5
        assert len(predict_receipts) == 2
        assert second["detections"][0]["geometry_skip_reason"] == "geometry_workers_busy"
        second_parts = second["timings_ms"]["post_parts"]
        assert second_parts["geometry_submitted"] == 0.0
        assert second_parts["geometry_inflight"] == 1.0

        # Once the old task returns, the next frame reaps it and recovers the
        # worker slot without ever admitting the stale result.
        release_geometry.set()
        deadline = time.monotonic() + 1.0
        while not next(iter(worker._geometry_inflight)).done():
            assert time.monotonic() < deadline
            time.sleep(0.005)
        raw.update(seq=3, stamp=10.2)
        third = worker.infer(raw)
        third_parts = third["timings_ms"]["post_parts"]
        assert third_parts["geometry_reaped"] == 1.0
        assert third_parts["geometry_completed"] == 1.0
        assert third["detections"][0]["geometry_skip_reason"] == "released_slow_geometry"
    finally:
        release_geometry.set()
        worker._geometry_executor.shutdown(wait=True, cancel_futures=True)


def test_explicit_yaw_does_not_evaluate_broken_optional_imu_rpy():
    pose = {"position": [5., 6., .3], "yaw": 0., "imu": {"rpy": []}}
    assert yolo._capture_pose_error(pose) is None
    rotation, offset = yolo._world_transform(pose, np.zeros(3), (0., 0., 0.))
    assert np.allclose(rotation, np.eye(3))
    assert np.allclose(offset, [5., 6., .3])
