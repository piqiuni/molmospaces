from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
ROS_SRC = REPO_ROOT / "Interactive-Nav-SG-nav" / "src"
RECORDER_SCRIPT_DIR = ROS_SRC / "explore_py_pkg" / "scripts"
for path in (ROS_SRC, RECORDER_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from record_explore_debug import (
    ExploreDebugRecorder,
    _AsyncArtifactWriter,
    _image_msg_to_rgb,
    _video_frame_export_policy,
)
from step_sync_image_cache import ExactStepImageCache


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
