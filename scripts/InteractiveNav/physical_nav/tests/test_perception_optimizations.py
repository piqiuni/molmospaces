"""Offline regression tests for the physical RGB-D perception hot paths."""

from __future__ import annotations

import pathlib
import sys
import threading
import base64
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import cv2


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physical_sensor_ros_bridge import (  # noqa: E402
    SensorRosBridge,
    _decode,
    _sample_depth_points,
)
from physical_yoloe_bridge import (  # noqa: E402
    _compact_mask_coordinates,
    _compact_web_report,
    _depth_projection_roi,
    _fast_quantile_pair,
    _geometry_lift_candidate,
    _interpolate_ordered_quantile,
    _materialize_selected_masks,
    _mask_uint8_view,
    _oriented_bounds,
    _project_depth_grid,
    _rgb_mask_to_depth,
    _prepare_instance_points,
    _quaternion_matrix,
    _transport_report_metadata,
)


def test_depth_projection_cache_reuses_rays_and_preserves_range_semantics():
    depth = np.asarray(
        [[1000, 0, 9000, 2000], [500, 1500, 2500, 3500]], dtype=np.uint16
    )
    intrinsics = {"fx": 2.0, "fy": 4.0, "cx": 1.0, "cy": 0.0}
    points, cache = _sample_depth_points(
        depth,
        intrinsics,
        stride=1,
        scale=0.001,
        max_depth_m=3.0,
        no_return_depth_m=3.25,
    )
    assert cache is not None
    # The zero-depth pixel is omitted.  The 9 m sample is represented as the
    # configured no-return ray, exactly as the pre-cache cloud path did.
    assert points.shape == (7, 3)
    assert np.isclose(points[:, 2].max(), 3.25)
    assert np.isclose(points[0, 0], -0.5)

    second, reused = _sample_depth_points(
        depth,
        intrinsics,
        stride=1,
        scale=0.001,
        max_depth_m=3.0,
        no_return_depth_m=3.25,
        projection_cache=cache,
    )
    assert reused is cache
    assert np.array_equal(second, points)


def test_depth_projection_cache_invalidates_on_stream_geometry_change():
    depth = np.ones((3, 4), dtype=np.uint16) * 1000
    base = {"fx": 4.0, "fy": 4.0, "cx": 1.5, "cy": 1.0}
    _, cache = _sample_depth_points(
        depth, base, stride=2, scale=0.001, max_depth_m=8.0, no_return_depth_m=8.05
    )
    changed = dict(base, cx=2.0)
    _, changed_cache = _sample_depth_points(
        depth,
        changed,
        stride=2,
        scale=0.001,
        max_depth_m=8.0,
        no_return_depth_m=8.05,
        projection_cache=cache,
    )
    assert changed_cache is not cache
    assert changed_cache[0] != cache[0]


def test_compact_mask_coordinates_is_bounded_and_round_trips_membership():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[2:12, 4:22] = 1
    rows, cols = _compact_mask_coordinates(mask, max_points=37)
    assert rows.shape == cols.shape
    assert rows.size <= 37
    assert np.all(mask[rows, cols] != 0)

    all_rows, all_cols = _compact_mask_coordinates(mask, max_points=1000)
    expected = int(np.count_nonzero(mask))
    assert all_rows.size == expected
    assert set(zip(all_rows.tolist(), all_cols.tolist())) == {
        (int(row), int(col)) for row, col in zip(*np.nonzero(mask))
    }


def test_bool_mask_uint8_view_is_zero_copy_and_geometry_equivalent():
    mask = np.zeros((32, 48), dtype=bool)
    mask[4:20, 7:31] = True
    view = _mask_uint8_view(mask)
    assert view.dtype == np.uint8
    assert np.shares_memory(view, mask)
    assert np.array_equal(view != 0, mask)
    noncontiguous = mask[:, ::2]
    converted = _mask_uint8_view(noncontiguous)
    assert converted.dtype == np.uint8
    assert np.array_equal(converted != 0, noncontiguous)


def test_parallel_geometry_lift_matches_single_worker_result():
    """Threaded candidates retain the exact RGB-D geometry contract."""
    mask = np.zeros((80, 100), dtype=np.float32)
    mask[12:68, 18:82] = 1.0
    depth = np.full(mask.shape, 2.0, dtype=np.float32)
    intr = {"width": 100, "height": 80, "fx": 80.0, "fy": 80.0, "cx": 50.0, "cy": 40.0}
    config = {
        "mask_component_min_area": 8,
        "point_stride": 2,
        "depth_scale": 1.0,
        "max_depth_m": 8.0,
        "min_valid_points": 8,
        "depth_band_lower_quantile": 0.02,
        "depth_band_upper_quantile": 0.98,
        "bbox_quantile_lower": 0.1,
        "bbox_quantile_upper": 0.9,
        "enable_euclidean_cluster": False,
    }
    args = (
        mask,
        depth,
        intr,
        intr,
        {"rotation": np.eye(3, dtype=np.float32).reshape(-1).tolist(), "translation": [0.0, 0.0, 0.0]},
        None,
        (0, 0, 100, 80),
        1.0,
        True,
        (80.0, 80.0, 50.0, 40.0),
        config,
        "chair",
        (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)),
    )
    serial = _geometry_lift_candidate(*args)
    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel = list(executor.map(lambda _: _geometry_lift_candidate(*args), range(4)))[0]
    assert serial["ok"] == parallel["ok"]
    assert serial.get("reason") == parallel.get("reason")
    for key in ("mask", "points", "values", "world"):
        assert np.array_equal(serial[key], parallel[key])
    for key in (
        "camera_center", "camera_size", "camera_obb_center", "camera_obb_size",
        "center", "size", "obb_center", "obb_size",
    ):
        assert np.allclose(serial[key], parallel[key])
    assert serial["camera_obb_orientation"] == parallel["camera_obb_orientation"]
    assert serial["obb_orientation"] == parallel["obb_orientation"]
    assert serial["obb_yaw"] == parallel["obb_yaw"]


def test_full_cloud_camera_obb_round_trips_through_world_transform():
    """The compact OBB contract preserves the full-cloud world geometry."""
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[12:68, 18:82] = 1
    depth = np.full(mask.shape, 2000, dtype=np.uint16)
    intr = {
        "width": 100, "height": 80, "fx": 80.0, "fy": 80.0,
        "cx": 50.0, "cy": 40.0,
    }
    config = {
        "mask_component_min_area": 8,
        "point_stride": 2,
        "depth_scale": 0.001,
        "max_depth_m": 8.0,
        "min_valid_points": 8,
        "depth_band_lower_quantile": 0.02,
        "depth_band_upper_quantile": 0.98,
        "bbox_quantile_lower": 0.1,
        "bbox_quantile_upper": 0.9,
        "enable_euclidean_cluster": False,
    }
    yaw = 0.37
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    translation = np.asarray([1.2, -0.7, 0.5], dtype=np.float32)
    result = _geometry_lift_candidate(
        mask,
        depth,
        intr,
        intr,
        {
            "rotation": np.eye(3, dtype=np.float32).reshape(-1).tolist(),
            "translation": [0.0, 0.0, 0.0],
        },
        None,
        (0, 0, 100, 80),
        0.001,
        True,
        (80.0, 80.0, 50.0, 40.0),
        config,
        "chair",
        (rotation, translation),
    )

    assert result["ok"]
    reconstructed_center = rotation @ result["camera_obb_center"] + translation
    reconstructed_axes = (
        rotation
        @ _quaternion_matrix(tuple(result["camera_obb_orientation"]))
    )
    world_axes = _quaternion_matrix(tuple(result["obb_orientation"]))
    assert np.allclose(reconstructed_center, result["obb_center"], atol=1e-5)
    assert np.allclose(reconstructed_axes, world_axes, atol=1e-5)
    assert np.allclose(result["camera_obb_size"], result["obb_size"])


def test_web_report_drops_heavy_geometry_without_mutating_ros_report():
    report = {
        "seq": 7,
        "stamp": 12.5,
        "transport_session": "sensor-session",
        "transport_connection": 3,
        "overlay_jpeg": "a-large-base64-image",
        "detections": [
            {
                "semantic_class": "door",
                "confidence": 0.91,
                "bbox": [1, 2, 30, 40],
                "world_box3d_center": [1.0, 2.0, 0.8],
                "mask": {"rows": [1, 2], "cols": [3, 4]},
                "world_segment_points_f32": "packed-cloud",
            }
        ],
    }
    compact = _compact_web_report(report)
    assert compact["web_compact"] is True
    assert compact["transport_session"] == "sensor-session"
    assert compact["transport_connection"] == 3
    assert "overlay_jpeg" not in compact
    assert compact["detections"] == [{
        "semantic_class": "door",
        "confidence": 0.91,
        "bbox": [1, 2, 30, 40],
        "world_box3d_center": [1.0, 2.0, 0.8],
    }]
    assert "overlay_jpeg" in report
    assert "mask" in report["detections"][0]


def test_yolo_report_transport_identity_comes_from_capture_context() -> None:
    raw = {
        "capture_timing": {
            "transport_session": "go2-source",
            "transport_connection": "4",
        }
    }
    assert _transport_report_metadata(raw) == {
        "transport_session": "go2-source",
        "transport_connection": 4,
    }
    assert _transport_report_metadata({"capture_timing": {}}) == {}
    assert _transport_report_metadata({
        "capture_timing": {
            "transport_session": "go2-source",
            "transport_connection": "invalid",
        }
    }) == {}


def test_uint8_mask_fast_path_is_equivalent_to_binary_normalization():
    """The no-copy uint8 path must preserve 2-D/3-D geometry exactly."""
    mask = np.zeros((64, 80), dtype=np.uint8)
    mask[8:48, 12:62] = 7  # non-zero values are all foreground to OpenCV
    depth = np.full(mask.shape, 1500, dtype=np.uint16)
    config = {
        "mask_component_min_area": 8,
        "point_stride": 2,
        "depth_scale": 0.001,
        "max_depth_m": 8.0,
        "min_valid_points": 8,
        "depth_band_lower_quantile": 0.02,
        "depth_band_upper_quantile": 0.98,
        "enable_euclidean_cluster": False,
    }
    fast = _prepare_instance_points(
        mask,
        depth,
        (80.0, 80.0, 40.0, 32.0),
        (12, 8, 62, 48),
        config,
    )
    normalized = _prepare_instance_points(
        (mask > 0).astype(np.uint8),
        depth,
        (80.0, 80.0, 40.0, 32.0),
        (12, 8, 62, 48),
        config,
    )
    assert fast is not None and normalized is not None
    assert np.array_equal(fast[0], normalized[0])
    assert np.array_equal(fast[1], normalized[1])
    assert np.array_equal(fast[2], normalized[2])


def test_fast_quantile_pair_matches_numpy_linear_quantiles():
    values = np.random.default_rng(4).normal(size=(257, 3)).astype(np.float32)
    lower, upper = _fast_quantile_pair(values, 0.02, 0.98, axis=0)
    expected = np.quantile(values, [0.02, 0.98], axis=0)
    assert np.allclose(lower, expected[0])
    assert np.allclose(upper, expected[1])


def test_quantile_interpolation_direct_sample_axis_matches_generic_axis_path():
    """The first-axis fast path must retain linear-quantile semantics."""
    values = np.random.default_rng(12).normal(size=(31, 5)).astype(np.float32)
    for axis in (0, 1):
        ordered = np.sort(values, axis=axis, kind="quicksort")
        for quantile in (0.0, 0.17, 0.5, 0.98, 1.0):
            actual = _interpolate_ordered_quantile(
                ordered, quantile, axis=axis
            )
            expected = np.quantile(values, quantile, axis=axis)
            assert np.allclose(actual, expected)


def test_oriented_bounds_reuses_supplied_mean_without_geometry_change():
    values = np.random.default_rng(21).normal(size=(257, 3)).astype(np.float32)
    config = {"bbox_quantile_lower": 0.02, "bbox_quantile_upper": 0.98}
    expected = _oriented_bounds(values, config)
    reused = _oriented_bounds(values, config, center=np.mean(values, axis=0))
    assert np.allclose(reused[0], expected[0])
    assert np.allclose(reused[1], expected[1])
    assert np.allclose(reused[2], expected[2])


def test_depth_projection_union_roi_is_pixel_equivalent_to_full_grid():
    """The ROI fast path must not alter RGB-to-native-depth mask selection."""
    depth = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48) + 900
    rgb_intr = {"width": 96, "height": 64, "fx": 74.0, "fy": 73.0, "cx": 48.0, "cy": 32.0}
    depth_intr = {"width": 48, "height": 32, "fx": 39.0, "fy": 39.5, "cx": 23.5, "cy": 15.5}
    extr = {"rotation": np.eye(3, dtype=np.float32).reshape(-1).tolist(), "translation": [0.025, -0.004, 0.0]}
    boxes = np.asarray([[8.0, 5.0, 45.0, 35.0], [52.0, 20.0, 90.0, 60.0]])
    roi = _depth_projection_roi(boxes, depth.shape, rgb_intr)
    full_maps = _project_depth_grid(
        depth.shape, (64, 96), rgb_intr, depth_intr, extr,
        depth_values=depth, depth_scale=0.001,
    )
    roi_maps = _project_depth_grid(
        depth.shape, (64, 96), rgb_intr, depth_intr, extr,
        depth_values=depth, depth_scale=0.001, projection_roi=roi,
    )
    assert all(np.array_equal(left, right) for left, right in zip(full_maps, roi_maps))
    mask = np.zeros((64, 96), dtype=np.uint8)
    mask[5:36, 8:46] = 1
    full = _rgb_mask_to_depth(
        mask, depth, rgb_intr, depth_intr, extr, full_maps, tuple(boxes[0]), 0.001
    )
    cropped = _rgb_mask_to_depth(
        mask, depth, rgb_intr, depth_intr, extr, roi_maps, tuple(boxes[0]), 0.001
    )
    assert np.array_equal(full, cropped)


def test_equal_shape_unaligned_stream_still_projects_with_baseline():
    """Equal RGB/depth dimensions do not imply aligned optical centres."""
    depth = np.ones((4, 4), dtype=np.float32)
    intr = {"width": 4, "height": 4, "fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0}
    extr = {"rotation": np.eye(3, dtype=np.float32).reshape(-1).tolist(), "translation": [1.0, 0.0, 0.0]}
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 1
    aligned = _rgb_mask_to_depth(
        mask, depth, intr, intr, extr, aligned_stream=True
    )
    unaligned = _rgb_mask_to_depth(
        mask, depth, intr, intr, extr, aligned_stream=False
    )
    assert aligned[1, 1]
    assert unaligned[1, 0]
    assert not unaligned[1, 1]


def test_yolo_waits_for_matching_rgb_depth_capture_stamp_before_lifting():
    """Interleaved ROS callbacks must never create a mixed RGB-D geometry pair."""
    # Avoid constructing the detector/model in this unit test.  The method
    # under test only needs the callback-owned latest buffers and calibration.
    from physical_yoloe_bridge import YoloeWorker

    worker = object.__new__(YoloeWorker)
    worker._frame_lock = threading.Lock()
    worker._rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    worker._depth = np.ones((2, 2), dtype=np.uint16)
    worker._rgb_stamp = 10.0
    worker._depth_stamp = 9.95
    worker._rgb_seq = 11
    worker._depth_seq = 10
    info = SimpleNamespace(
        width=2,
        height=2,
        K=[1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0],
        D=[],
        distortion_model="",
        header=SimpleNamespace(frame_id="color"),
    )
    depth_info = SimpleNamespace(
        width=2,
        height=2,
        K=[1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0],
        D=[],
        distortion_model="",
        header=SimpleNamespace(frame_id="depth"),
    )
    worker._camera_info = info
    worker._depth_camera_info = depth_info
    worker._depth_scale = 0.001
    worker._depth_to_color_extrinsics = {}
    worker._last_frame_key = None
    worker._pair_mismatch_count = 0
    worker._pair_mismatch_last = None
    worker._telemetry_lock = threading.Lock()
    worker._telemetry_history = []

    assert worker._latest_raw_frame() is None
    assert worker._pair_mismatch_count == 1

    worker._depth_stamp = 10.0
    raw = worker._latest_raw_frame()
    assert raw is not None
    assert raw["rgb_seq"] == 11
    assert raw["depth_seq"] == 10
    assert raw["seq"] == 11

    # The inference loop must not repeatedly lift the same capture while ROS
    # is waiting for the next sensor frame.
    assert worker._latest_raw_frame() is None


def test_yolo_telemetry_history_is_scoped_to_transport_generation():
    """Reconnects must not reuse poses from an old source clock/session."""
    from physical_yoloe_bridge import YoloeWorker

    worker = object.__new__(YoloeWorker)
    worker._telemetry_lock = threading.Lock()
    # Simulate a legacy packet arriving before the bridge starts tagging
    # telemetry.  The first tagged packet must invalidate this unknown clock.
    worker._telemetry_history = [(100.0, {"yaw": 0.1})]
    worker._telemetry_transport_session = None
    worker._telemetry_transport_connection = -1
    worker._telemetry_retired_sessions = set()

    def packet(stamp, session=None, connection=0, yaw=0.0):
        telemetry = {"received_at": stamp, "yaw": yaw}
        if session is not None:
            telemetry["_transport_session"] = session
            telemetry["_transport_connection"] = connection
        return SimpleNamespace(
            data=json.dumps({"stamp": stamp, "telemetry": telemetry})
        )

    worker._telemetry_callback(packet(100.0, "session-old", 1, 0.2))
    worker._telemetry_callback(packet(1.0, "session-new", 1, 1.2))
    assert len(worker._telemetry_history) == 1
    assert worker._telemetry_at(1.0)["yaw"] == 1.2

    # Delayed packets from the retired generation and unmarked legacy packets
    # are rejected once a tagged transport is active.
    worker._telemetry_callback(packet(1.1, "session-old", 1, 9.0))
    worker._telemetry_callback(packet(1.2, None, 0, 8.0))
    assert len(worker._telemetry_history) == 1
    assert worker._telemetry_at(1.1)["yaw"] == 1.2

    # A fresh connection in the same process can also restart the source
    # clock; it starts a clean nearest-neighbour history.
    worker._telemetry_callback(packet(2.0, "session-new", 2, 2.0))
    assert len(worker._telemetry_history) == 1
    assert worker._telemetry_at(2.0)["yaw"] == 2.0


def test_mask_materialization_copies_only_selected_instances():
    masks = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
    selected = _materialize_selected_masks(masks, [3, 1, 3])
    assert set(selected) == {1, 3}
    assert np.array_equal(selected[1], masks[1])
    assert np.array_equal(selected[3], masks[3])


def test_cloud_slot_rejects_retired_connection_not_newer_same_stream_frame():
    bridge = object.__new__(SensorRosBridge)
    bridge._packet_lock = threading.Lock()
    bridge._last_published_bridge_seq = 11
    bridge._latest_packet = {"_bridge_seq": 12}
    bridge._active_transport_session = "camera"
    bridge._active_transport_connection = 2
    assert bridge._cloud_generation_is_current(("camera", 1)) is False
    assert bridge._cloud_generation_is_current(("old-camera", 2)) is False
    assert bridge._cloud_generation_is_current(("camera", 2)) is True


def test_live_bgr_decode_matches_rgb_contract_with_one_channel_swap():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :, 0] = 240
    rgb[:, :, 2] = 20
    ok, encoded = cv2.imencode(
        ".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 100]
    )
    assert ok
    packet_value = {
        "encoding": "jpeg",
        "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }
    decoded_rgb = _decode(packet_value)
    decoded_bgr = _decode(packet_value, True)
    assert np.allclose(decoded_rgb[:, :, ::-1], decoded_bgr, atol=2)
