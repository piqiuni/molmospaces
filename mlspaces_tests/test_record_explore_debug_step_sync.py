from __future__ import annotations

import json
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ROS_SRC = REPO_ROOT / "Interactive-Nav-SG-nav" / "src"
RECORDER_SCRIPT_DIR = ROS_SRC / "explore_py_pkg" / "scripts"
for path in (ROS_SRC, RECORDER_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String

from record_explore_debug import (
    ExploreDebugRecorder,
    _FrozenVideoGrid,
    _AsyncArtifactWriter,
    _freeze_video_grid,
    _image_msg_to_rgb,
    _known_world_bounds_from_grid,
    _video_grid_render_rgb,
    _video_frame_export_policy,
    _world_to_cell,
)
from step_sync_image_cache import CachedStepImage, ExactStepImageCache


def _recorder_stub() -> tuple[ExploreDebugRecorder, list[tuple]]:
    recorder = ExploreDebugRecorder.__new__(ExploreDebugRecorder)
    recorder.args = SimpleNamespace(
        first_person_video_capture_mode="step",
        video_step_sync_topic="/molmo_spaces/step_sync",
    )
    recorder.shutting_down = False
    recorder.lock = threading.RLock()
    recorder.last_recorded_image_key = None
    recorder.last_recorded_image_stamp_ns = None
    recorder.last_step_sync_key = None
    recorder.last_source_image_seq = None
    recorder.image_callback_count = 0
    recorder.step_sync_count = 0
    recorder.step_sync_capture_every = 1
    recorder.step_sync_capture_count = 0
    recorder.step_sync_skipped_count = 0
    recorder.step_sync_image_match_count = 0
    recorder.step_sync_image_reuse_count = 0
    recorder.step_sync_placeholder_count = 0
    recorder.debug_step = 0
    recorder.latest_image = None
    recorder.latest_image_step = 0
    recorder.last_image_wall_time = 0.0
    recorder.step_sync_placeholder_width = 4
    recorder.step_sync_placeholder_height = 2
    recorder.step_sync_placeholder_rgb = bytes(4 * 2 * 3)
    recorder.step_sync_image_wait_sec = 0.5
    recorder.step_sync_image_max_stamp_delta_ns = 5_000_000
    recorder.step_sync_image_fallback_max_age_ns = 3_000_000_000
    recorder.step_sync_image_cache_size = 128
    recorder.step_sync_image_cache = ExactStepImageCache(
        max_size=recorder.step_sync_image_cache_size
    )
    recorder._capture_video_snapshot_locked = lambda stamp: {"capture_stamp": stamp}
    jobs: list[tuple] = []
    recorder._enqueue_video_frame = lambda job: jobs.append(job) or True
    return recorder, jobs


def _render_queue_recorder_stub(overflow: str) -> ExploreDebugRecorder:
    recorder = ExploreDebugRecorder.__new__(ExploreDebugRecorder)
    recorder.args = SimpleNamespace(video_frame_queue_overflow=overflow)
    recorder.shutting_down = False
    recorder.video_frame_jobs = queue.Queue(maxsize=2)
    recorder.video_frame_enqueue_lock = threading.Lock()
    recorder.video_frame_jobs_dropped = 0
    recorder.video_frame_jobs_dropped_oldest = 0
    recorder.video_frame_jobs_dropped_newest = 0
    recorder.video_frame_jobs_dropped_oldest_steps = []
    return recorder


def _render_job(step_id: int) -> tuple:
    return (4, 2, bytes(4 * 2 * 3), 0.0, step_id, {"step": step_id})


def _occupancy_grid(width: int = 8, height: int = 4) -> OccupancyGrid:
    grid = OccupancyGrid()
    grid.header.frame_id = "map"
    grid.info.width = width
    grid.info.height = height
    grid.info.resolution = 0.1
    grid.info.origin.position.x = -1.0
    grid.info.origin.position.y = 2.0
    grid.info.origin.orientation.w = 1.0
    grid.data = [-1] * (width * height)
    for y in range(1, height):
        for x in range(1, width):
            grid.data[y * width + x] = 0 if (x + y) % 3 else 100
    return grid


def _video_history_recorder_stub() -> ExploreDebugRecorder:
    recorder = ExploreDebugRecorder.__new__(ExploreDebugRecorder)
    recorder.args = SimpleNamespace(
        video_snapshot_grid_max_dim=4,
        video_snapshot_jpeg_quality=90,
    )
    recorder._retain_video_state_history = True
    recorder.lock = threading.RLock()
    recorder.shutting_down = False
    recorder._occupancy_pairing_enabled = False
    recorder.debug_step = 7
    recorder.latest_grid = None
    recorder.latest_grid_video_grid = None
    recorder.latest_grid_video_rgb = None
    recorder.latest_grid_video_stamp = 0.0
    recorder.latest_grid_wall_time = 0.0
    recorder.latest_grid_step = 0
    recorder.grid_video_history = deque(maxlen=2)
    recorder.latest_global_costmap = None
    recorder.latest_global_costmap_video_grid = None
    recorder.latest_global_costmap_video_rgb = None
    recorder.latest_global_costmap_video_stamp = 0.0
    recorder.latest_global_costmap_wall_time = 0.0
    recorder.latest_global_costmap_step = 0
    recorder.global_costmap_video_history = deque(maxlen=2)
    recorder.latest_local_costmap = None
    recorder.latest_local_costmap_video_grid = None
    recorder.latest_local_costmap_video_rgb = None
    recorder.latest_local_costmap_video_stamp = 0.0
    recorder.latest_local_costmap_wall_time = 0.0
    recorder.latest_local_costmap_step = 0
    recorder.local_costmap_video_history = deque(maxlen=2)
    recorder.latest_scene_id_grid = None
    recorder.latest_scene_id_grid_rgb = None
    recorder.latest_scene_id_grid_stamp = 0.0
    recorder.latest_scene_id_grid_step = 0
    recorder.scene_id_grid_history = deque(maxlen=2)
    recorder.room_segment_callback_count = 0
    recorder.latest_room_segment_valid_cell_count = 0
    recorder.latest_room_segment_unique_ids = []
    return recorder


def _rgb_message(seq: int, stamp_sec: float, color: tuple[int, int, int]) -> Image:
    msg = Image()
    msg.header.seq = seq
    msg.header.stamp = rospy.Time.from_sec(stamp_sec)
    msg.width = 8
    msg.height = 6
    msg.encoding = "rgb8"
    msg.step = msg.width * 3
    msg.data = bytes(color) * (msg.width * msg.height)
    return msg


def test_rgb8_decode_uses_real_pixels_without_per_pixel_transformation() -> None:
    msg = _rgb_message(3, 10.0, (210, 35, 20))
    width, height, rgb = _image_msg_to_rgb(msg)
    assert (width, height) == (8, 6)
    assert rgb == bytearray(bytes((210, 35, 20)) * 48)


def test_ros_bridge_image_before_sync_contract_survives_callback_reordering() -> None:
    recorder, jobs = _recorder_stub()
    stamp_sec = 1_785_213_882.615378
    sync = String(
        data=json.dumps({"step_index": 17, "stamp_sec": stamp_sec})
    )

    sync_thread = threading.Thread(target=recorder.step_sync_callback, args=(sync,))
    sync_thread.start()
    time.sleep(0.03)
    recorder.image_callback(_rgb_message(17, stamp_sec, (210, 35, 20)))
    sync_thread.join(timeout=1.0)

    assert not sync_thread.is_alive()
    assert len(jobs) == 1
    width, height, rgb, _stamp, source_seq, snapshot = jobs[0]
    assert (width, height, source_seq) == (8, 6, 17)
    assert rgb == bytes((210, 35, 20)) * 48
    assert snapshot["camera_source"] == "exact_image"
    assert recorder.step_sync_image_match_count == 1
    assert recorder.step_sync_placeholder_count == 0


def test_missing_exact_callback_uses_recent_real_image_not_placeholder() -> None:
    recorder, jobs = _recorder_stub()
    recorder.image_callback(_rgb_message(8, 20.0, (90, 120, 150)))
    recorder.step_sync_callback(
        String(data=json.dumps({"step_index": 9, "stamp_sec": 21.0}))
    )

    assert len(jobs) == 1
    assert jobs[0][2] == bytes((90, 120, 150)) * 48
    assert jobs[0][5]["camera_source"] == "nearest_image"
    assert recorder.step_sync_image_match_count == 1
    assert recorder.step_sync_placeholder_count == 0


def test_image_cache_reuses_latest_real_image_across_nine_second_gap() -> None:
    cache = ExactStepImageCache(max_size=8)
    old = CachedStepImage(
        source_seq=4,
        stamp_ns=100_000_000_000,
        stamp=100.0,
        width=2,
        height=1,
        rgb=bytes((10, 20, 30, 40, 50, 60)),
    )
    exact = CachedStepImage(
        source_seq=5,
        stamp_ns=108_000_000_000,
        stamp=108.0,
        width=2,
        height=1,
        rgb=bytes((70, 80, 90, 100, 110, 120)),
    )
    cache.put(old)
    cache.put(exact)

    # The exact image still wins before any long-gap fallback is considered.
    exact_selection = cache.wait_select(
        source_seq=5,
        stamp_ns=108_000_000_000,
        timeout_sec=0.0,
        max_stamp_delta_ns=5_000_000,
        fallback_max_age_ns=12_000_000_000,
    )
    assert exact_selection is not None
    assert exact_selection.source == "exact_image"
    assert exact_selection.frame == exact

    # A 9.5-second RGB outage must retain the last real image, not go black.
    fallback_selection = cache.wait_select(
        source_seq=6,
        stamp_ns=117_500_000_000,
        timeout_sec=0.0,
        max_stamp_delta_ns=5_000_000,
        fallback_max_age_ns=12_000_000_000,
    )
    assert fallback_selection is not None
    assert fallback_selection.source == "latest_image"
    assert fallback_selection.reused is True
    assert fallback_selection.frame == exact


def test_eighty_images_survive_delayed_sync_backlog_without_placeholders() -> None:
    recorder, jobs = _recorder_stub()
    base_stamp = 1_785_213_800.0
    for seq in range(80):
        recorder.image_callback(
            _rgb_message(seq, base_stamp + seq * 0.2, (seq, 50, 100))
        )

    assert recorder.step_sync_image_cache.active_size == 80
    for seq in range(80):
        recorder.step_sync_callback(
            String(
                data=json.dumps(
                    {
                        "step_index": seq,
                        "stamp_sec": base_stamp + seq * 0.2,
                    }
                )
            )
        )

    assert len(jobs) == 80
    assert recorder.step_sync_count == 80
    assert recorder.step_sync_image_match_count == 80
    assert recorder.step_sync_placeholder_count == 0
    assert recorder.step_sync_image_reuse_count == 0
    assert recorder.step_sync_image_cache.active_size == 0
    assert all(job[5]["camera_source"] == "exact_image" for job in jobs)


def test_step_sync_sampling_keeps_raw_progress_without_rendering_every_marker() -> None:
    recorder, jobs = _recorder_stub()
    recorder.step_sync_capture_every = 2
    base_stamp = 1_785_213_900.0
    for seq in range(4):
        recorder.image_callback(
            _rgb_message(seq, base_stamp + seq * 0.2, (seq, 50, 100))
        )
        recorder.step_sync_callback(
            String(
                data=json.dumps(
                    {
                        "step_index": seq,
                        "stamp_sec": base_stamp + seq * 0.2,
                    }
                )
            )
        )

    assert recorder.step_sync_count == 4
    assert recorder.step_sync_capture_count == 2
    assert recorder.step_sync_skipped_count == 2
    assert recorder.step_sync_image_match_count == 2
    assert recorder.step_sync_placeholder_count == 0
    assert [job[4] for job in jobs] == [0, 2]
    assert [job[5]["callback_index"] for job in jobs] == [1, 3]
    assert [job[5]["capture_index"] for job in jobs] == [1, 2]


def test_render_queue_drop_oldest_keeps_newest_snapshot_and_balances_tasks() -> None:
    recorder = _render_queue_recorder_stub("drop_oldest")
    assert recorder._enqueue_video_frame(_render_job(10)) is True
    assert recorder._enqueue_video_frame(_render_job(11)) is True
    assert recorder._enqueue_video_frame(_render_job(12)) is True

    queued = [recorder.video_frame_jobs.get_nowait(), recorder.video_frame_jobs.get_nowait()]
    assert [job[4] for job in queued] == [11, 12]
    for _job in queued:
        recorder.video_frame_jobs.task_done()
    recorder.video_frame_jobs.join()

    assert recorder.video_frame_jobs_dropped == 1
    assert recorder.video_frame_jobs_dropped_oldest == 1
    assert recorder.video_frame_jobs_dropped_newest == 0
    assert recorder.video_frame_jobs_dropped_oldest_steps == [10]


def test_render_queue_drop_still_rejects_newest_snapshot() -> None:
    recorder = _render_queue_recorder_stub("drop")
    assert recorder._enqueue_video_frame(_render_job(10)) is True
    assert recorder._enqueue_video_frame(_render_job(11)) is True
    assert recorder._enqueue_video_frame(_render_job(12)) is False

    queued = [recorder.video_frame_jobs.get_nowait(), recorder.video_frame_jobs.get_nowait()]
    assert [job[4] for job in queued] == [10, 11]
    for _job in queued:
        recorder.video_frame_jobs.task_done()
    recorder.video_frame_jobs.join()

    assert recorder.video_frame_jobs_dropped == 1
    assert recorder.video_frame_jobs_dropped_oldest == 0
    assert recorder.video_frame_jobs_dropped_newest == 1


def test_frozen_video_grid_keeps_scaled_transform_and_jpeg_not_source_map() -> None:
    source = _occupancy_grid()
    rgb = np.full((4, 8, 3), (30, 180, 90), dtype=np.uint8)
    frozen = _freeze_video_grid(
        source,
        rgb,
        max_dimension=4,
        jpeg_quality=90,
        known_world_bounds=_known_world_bounds_from_grid(source),
    )

    assert isinstance(frozen, _FrozenVideoGrid)
    assert frozen is not source
    assert frozen.data is None
    assert frozen.header.frame_id == "map"
    assert (frozen.info.width, frozen.info.height) == (4, 2)
    assert frozen.info.resolution == 0.2
    assert frozen.known_world_bounds is not None
    assert len(frozen.rgb_jpeg) > 20
    decoded = _video_grid_render_rgb(frozen, None)
    assert decoded is not None
    assert decoded.shape == (2, 4, 3)
    # The proxy spans the same world extent despite reducing 8 cells to 4.
    assert _world_to_cell(frozen, -0.31, 2.31) == (3, 1)


def test_frozen_occupancy_proxy_crops_in_metric_space_before_downsampling() -> None:
    source = _occupancy_grid(width=100, height=80)
    source.data = [-1] * (100 * 80)
    for y in range(30, 50):
        for x in range(40, 60):
            source.data[y * 100 + x] = 0
    rgb = np.full((80, 100, 3), (178, 178, 178), dtype=np.uint8)
    rgb[80 - 50 : 80 - 30, 40:60] = (248, 248, 245)
    known_bounds = _known_world_bounds_from_grid(source)

    compact = _freeze_video_grid(
        source,
        rgb,
        max_dimension=20,
        jpeg_quality=90,
        known_world_bounds=known_bounds,
        content_cell_bounds=(40, 30, 60, 50),
        visual_crop_margin_m=1.0,
        categorical=True,
        image_encoding="png",
    )
    detailed = _freeze_video_grid(
        source,
        rgb,
        max_dimension=40,
        jpeg_quality=90,
        known_world_bounds=known_bounds,
        content_cell_bounds=(40, 30, 60, 50),
        visual_crop_margin_m=1.0,
        categorical=True,
        image_encoding="png",
    )

    # Known content is 2 m square; both frozen resolutions retain exactly the
    # same 1 m perimeter on all sides rather than a proxy-pixel margin.
    assert (compact.info.width, compact.info.height) == (20, 20)
    assert compact.info.resolution == 0.2
    assert (detailed.info.width, detailed.info.height) == (40, 40)
    assert detailed.info.resolution == 0.1
    assert compact.info.width * compact.info.resolution == 4.0
    assert detailed.info.width * detailed.info.resolution == 4.0
    assert compact.info.origin.position.x == 2.0
    assert compact.info.origin.position.y == 4.0
    assert compact.image_encoding == "png"
    assert _world_to_cell(compact, 3.55, 5.55) == (7, 7)
    assert _world_to_cell(detailed, 3.55, 5.55) == (15, 15)


def test_categorical_frozen_proxy_uses_lossless_palette_not_jpeg_blending() -> None:
    source = _occupancy_grid(width=8, height=8)
    palette = np.asarray(
        [(178, 178, 178), (248, 248, 245), (28, 30, 32), (112, 36, 170)],
        dtype=np.uint8,
    )
    rgb = np.empty((8, 8, 3), dtype=np.uint8)
    for y in range(8):
        for x in range(8):
            rgb[y, x] = palette[(x + y) % len(palette)]
    frozen = _freeze_video_grid(
        source,
        rgb,
        max_dimension=4,
        jpeg_quality=1,
        categorical=True,
        image_encoding="png",
    )
    decoded = _video_grid_render_rgb(frozen, None)

    assert frozen.image_encoding == "png"
    assert decoded is not None
    decoded_colors = {tuple(pixel) for pixel in decoded.reshape((-1, 3))}
    assert decoded_colors <= {tuple(color) for color in palette}


def test_video_panel_metric_margin_does_not_depend_on_proxy_pixels() -> None:
    source = _occupancy_grid(width=100, height=80)
    source.data = [-1] * (100 * 80)
    for y in range(30, 50):
        for x in range(40, 60):
            source.data[y * 100 + x] = 0
    rgb = np.full((80, 100, 3), (178, 178, 178), dtype=np.uint8)
    rgb[80 - 50 : 80 - 30, 40:60] = (248, 248, 245)
    known_bounds = _known_world_bounds_from_grid(source)

    def frozen_at(max_dimension: int):
        return _freeze_video_grid(
            source,
            rgb,
            max_dimension=max_dimension,
            jpeg_quality=90,
            known_world_bounds=known_bounds,
            content_cell_bounds=(40, 30, 60, 50),
            # Retain a 2 m source perimeter, then render a 1 m perimeter.
            visual_crop_margin_m=2.0,
            categorical=True,
            image_encoding="png",
        )

    def render_bbox(frozen: _FrozenVideoGrid):
        recorder = ExploreDebugRecorder.__new__(ExploreDebugRecorder)
        recorder.args = SimpleNamespace(
            map_frame="map",
            odom_frame="map",
            video_map_crop_margin_px=25,
            frontier_check_radius_m=1.0,
            plan_goal_match_tolerance_m=1.0,
        )
        recorder.latest_pose = None
        recorder.latest_global_plan = None
        recorder.latest_local_global_plan = None
        recorder.latest_local_plan = None
        recorder.latest_image_step = 0
        recorder.video_map_bbox = None
        panel = recorder._render_video_map_panel_locked(
            100,
            100,
            grid=frozen,
            base=_video_grid_render_rgb(frozen, None),
            title="OCC",
            draw_frontiers=False,
            draw_global_plan=False,
            draw_local_global_plan=False,
            draw_local_plan=False,
            draw_goal=False,
            pose=None,
            goal_xy=(-0.5, 2.2),
            goal_yaw=0.0,
            trajectory=[],
            image_step=1,
            crop_margin_m=1.0,
            world_bounds=frozen.known_world_bounds,
        )
        assert panel is not None
        return recorder.video_map_bbox

    compact = frozen_at(30)
    detailed = frozen_at(60)
    compact_bbox = render_bbox(compact)
    detailed_bbox = render_bbox(detailed)

    assert compact_bbox is not None and detailed_bbox is not None
    # Both left margins represent 1.0 m, even though they use 5 and 10
    # proxy pixels respectively.  The legacy 25px value is intentionally not
    # involved when a metric margin is supplied.
    assert compact_bbox[0] * compact.info.resolution == 1.0
    assert detailed_bbox[0] * detailed.info.resolution == 1.0


def test_costmap_world_bounds_replace_stale_map_wide_bbox() -> None:
    """A costmap can use the OCC viewport instead of accumulated plan bounds."""

    source = _occupancy_grid(width=100, height=80)
    rgb = np.full((80, 100, 3), (248, 248, 245), dtype=np.uint8)
    recorder = ExploreDebugRecorder.__new__(ExploreDebugRecorder)
    recorder.args = SimpleNamespace(
        map_frame="map",
        odom_frame="map",
        video_map_crop_margin_px=90,
        frontier_check_radius_m=1.0,
        plan_goal_match_tolerance_m=1.0,
    )
    recorder.latest_pose = None
    recorder.latest_global_plan = None
    recorder.latest_local_global_plan = None
    recorder.latest_local_plan = None
    recorder.latest_image_step = 0
    # This represents the old persistent map-wide crop accumulated before the
    # current OCC view is supplied.
    recorder.video_global_costmap_bbox = (0, 0, 99, 79)
    occ_world_bounds = (2.0, 4.0, 4.0, 6.0)

    panel = recorder._render_video_map_panel_locked(
        100,
        100,
        grid=source,
        base=rgb,
        title="GLOBAL COSTMAP",
        bbox_attr="video_global_costmap_bbox",
        draw_frontiers=False,
        draw_global_plan=False,
        draw_local_global_plan=False,
        draw_local_plan=False,
        draw_goal=False,
        pose=None,
        # Supplying an inert goal avoids consulting the recorder's live goal
        # state; it is not drawn in this regression test.
        goal_xy=(-1.0, 2.0),
        goal_yaw=0.0,
        trajectory=[],
        crop_margin_m=1.0,
        world_bounds=occ_world_bounds,
    )

    assert panel is not None
    # The 2 m OCC bounds plus 1 m physical margin maps to x pixels [20, 60]
    # and flipped-y pixels [29, 69]; the prior full-map bbox is not retained.
    assert recorder.video_global_costmap_bbox == (20, 29, 60, 69)


def test_video_map_histories_store_proxies_for_all_four_grid_topics() -> None:
    recorder = _video_history_recorder_stub()
    source = _occupancy_grid()

    recorder.occupancy_callback(source)
    recorder.global_costmap_callback(source)
    recorder.local_costmap_callback(source)
    recorder.scene_id_grid_callback(source)

    histories = (
        recorder.grid_video_history,
        recorder.global_costmap_video_history,
        recorder.local_costmap_video_history,
        recorder.scene_id_grid_history,
    )
    for history in histories:
        frozen = history[-1][1]
        encoded_rgb = history[-1][2]
        assert isinstance(frozen, _FrozenVideoGrid)
        assert frozen.data is None
        assert isinstance(encoded_rgb, bytes)
        assert encoded_rgb == frozen.rgb_jpeg
        assert _video_grid_render_rgb(frozen, encoded_rgb) is not None

    # Full ROS messages are retained only as the one current diagnostic map,
    # never in a history entry captured by pending video jobs.
    assert recorder.latest_grid is source
    assert recorder.latest_global_costmap is not source


def test_global_costmap_video_proxy_keeps_native_lossless_grid() -> None:
    recorder = _video_history_recorder_stub()
    source = _occupancy_grid(width=8, height=4)

    recorder.global_costmap_callback(source)
    recorder.local_costmap_callback(source)

    global_frozen = recorder.global_costmap_video_history[-1][1]
    local_frozen = recorder.local_costmap_video_history[-1][1]
    assert isinstance(global_frozen, _FrozenVideoGrid)
    assert isinstance(local_frozen, _FrozenVideoGrid)
    # Global costmaps explain global-plan clearance, so their six-panel proxy
    # must retain the original cells instead of the normal bounded JPEG proxy.
    assert (global_frozen.info.width, global_frozen.info.height) == (8, 4)
    assert global_frozen.info.resolution == source.info.resolution
    assert global_frozen.image_encoding == "png"
    # The rolling local map remains bounded for the render queue.
    assert (local_frozen.info.width, local_frozen.info.height) == (4, 2)
    assert local_frozen.image_encoding == "jpeg"


def test_frozen_grid_renders_path_and_room_panels_without_cell_payload() -> None:
    source = _occupancy_grid()
    frozen = _freeze_video_grid(
        source,
        np.full((4, 8, 3), (80, 140, 220), dtype=np.uint8),
        max_dimension=4,
        jpeg_quality=90,
        known_world_bounds=_known_world_bounds_from_grid(source),
    )
    rgb = _video_grid_render_rgb(frozen, None)
    recorder = ExploreDebugRecorder.__new__(ExploreDebugRecorder)
    recorder.args = SimpleNamespace(
        map_frame="map",
        odom_frame="map",
        video_map_crop_margin_px=1,
        frontier_check_radius_m=1.0,
        plan_goal_match_tolerance_m=1.0,
    )
    recorder.latest_pose = None
    recorder.latest_global_plan = None
    recorder.latest_local_global_plan = None
    recorder.latest_local_plan = None
    recorder.latest_image_step = 0
    recorder.video_map_bbox = None

    map_panel = recorder._render_video_map_panel_locked(
        80,
        40,
        grid=frozen,
        base=rgb,
        title="OCC",
        draw_frontiers=True,
        draw_global_plan=True,
        draw_local_global_plan=False,
        draw_local_plan=False,
        draw_goal=False,
        pose=None,
        goal_xy=(-0.5, 2.2),
        goal_yaw=0.0,
        trajectory=[],
        global_plan={
            "frame_id": "map",
            "poses": [(-0.9, 2.1, 0.0), (-0.5, 2.2, 0.0)],
        },
        image_step=1,
        world_bounds=frozen.known_world_bounds,
    )
    room_panel = recorder._render_room_segment_panel_locked(
        80,
        40,
        None,
        occupancy_grid=frozen,
        occupancy_rgb=rgb,
        scene_grid=frozen,
        scene_rgb=rgb,
        graph={},
        observed_instance_ids=set(),
        semantic_selection={},
        image_step=1,
        world_bounds=frozen.known_world_bounds,
    )

    assert map_panel is not None and map_panel.shape == (40, 80, 3)
    assert room_panel is not None and room_panel.shape == (40, 80, 3)


def test_runtime_mp4_does_not_require_per_step_png_exports() -> None:
    assert _video_frame_export_policy(
        runtime_video_encode=True,
        artifact_writer_available=True,
        save_panel_frames=False,
        save_composite_frames=False,
    ) == (False, False)
    assert _video_frame_export_policy(
        runtime_video_encode=False,
        artifact_writer_available=True,
        save_panel_frames=False,
        save_composite_frames=False,
    ) == (False, True)
    assert _video_frame_export_policy(
        runtime_video_encode=True,
        artifact_writer_available=False,
        save_panel_frames=False,
        save_composite_frames=False,
    ) == (False, True)
    assert _video_frame_export_policy(
        runtime_video_encode=True,
        artifact_writer_available=True,
        save_panel_frames=True,
        save_composite_frames=True,
    ) == (True, True)


def test_runtime_video_submission_backpressures_instead_of_dropping() -> None:
    writer = _AsyncArtifactWriter.__new__(_AsyncArtifactWriter)
    writer.video_jobs = __import__("queue").Queue(maxsize=1)
    writer.video_jobs.put(("occupied", Path("occupied.mp4"), object()))
    writer.submitted_video_jobs = 0
    writer.video_queue_peak = 0
    writer.dropped_jobs = 0

    class CopyableFrame:
        def copy(self):
            return self

    thread = threading.Thread(
        target=writer.submit_video,
        args=("first_person", Path("video.mp4"), CopyableFrame()),
    )
    thread.start()
    time.sleep(0.03)
    assert thread.is_alive()
    assert writer.dropped_jobs == 0

    writer.video_jobs.get_nowait()
    writer.video_jobs.task_done()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert writer.submitted_video_jobs == 1
    assert writer.video_jobs.qsize() == 1
