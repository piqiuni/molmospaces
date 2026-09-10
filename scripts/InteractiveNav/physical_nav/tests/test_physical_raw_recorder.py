import base64
import json
import pathlib
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physical_protocol import image_packet  # noqa: E402
from physical_raw_recorder import PhysicalRawRecorder  # noqa: E402
from showcase_pages import DARK_SHOWCASE_HTML  # noqa: E402


class PhysicalRawRecorderTests(unittest.TestCase):
    def test_inactive_ros_map_does_not_materialize_or_enqueue_payload(self):
        class ExplodesOnDeepcopy:
            def __deepcopy__(self, _memo):
                raise AssertionError("inactive map payload was deep-copied")

        with tempfile.TemporaryDirectory() as temporary:
            recorder = PhysicalRawRecorder(temporary)
            value = {
                "width": 1,
                "height": 1,
                "resolution": 0.1,
                "data": [ExplodesOnDeepcopy()],
            }

            self.assertFalse(recorder.record_ros_state("occupancy", value))
            self.assertEqual(recorder._receipt_counters, {})
            self.assertIsNone(recorder._queue)

    def test_inactive_common_event_does_not_deepcopy_or_enqueue_payload(self):
        class ExplodesOnDeepcopy:
            def __deepcopy__(self, _memo):
                raise AssertionError("inactive event payload was deep-copied")

        with tempfile.TemporaryDirectory() as temporary:
            recorder = PhysicalRawRecorder(temporary)

            self.assertFalse(
                recorder.record_event("large_state", ExplodesOnDeepcopy())
            )
            self.assertIsNone(recorder._queue)

    def _packet(self):
        rgb = np.zeros((24, 32, 3), dtype=np.uint8)
        ok, rgb_encoded = cv2.imencode(".jpg", rgb)
        self.assertTrue(ok)
        depth = np.full((24, 32), 1000, dtype=np.uint16)
        ok, depth_encoded = cv2.imencode(".png", depth)
        self.assertTrue(ok)
        return image_packet(
            seq=3,
            stamp=1.2,
            rgb_jpeg=bytes(rgb_encoded),
            depth_png=bytes(depth_encoded),
            width=32,
            height=24,
            camera_frame="d435i_color_optical_frame",
            depth_scale=0.001,
            intrinsics={"fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 12.0},
            color_depth_sync_ms=0.2,
        )

    def test_session_persists_raw_panels_right_state_and_phone_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = PhysicalRawRecorder(temporary, queue_size=64)
            started = recorder.start(
                label="showcase-dark",
                metadata={"sections": {"perception": "panel1/camera"}, "audio_source": "phone"},
            )
            self.assertTrue(started["active"])
            packet = self._packet()
            packet["capture_timing"] = {
                "version": 1,
                "timestamp_kind": "sensor_exposure_midpoint",
                "stamp_source": "device_clock_estimated",
                "dequeue_age_s": 0.02,
            }
            self.assertTrue(recorder.record_sensor_packet(packet))
            self.assertTrue(
                recorder.record_ros_state(
                    "occupancy",
                    {
                        "width": 4,
                        "height": 3,
                        "resolution": 0.05,
                        "origin": {"x": -0.1, "y": -0.1},
                        "frame_id": "tf_frame_map",
                        "data": [-1] * 12,
                    },
                )
            )
            ok, mllm_image = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
            self.assertTrue(ok)
            image_data = "data:image/jpeg;base64," + base64.b64encode(bytes(mllm_image)).decode()
            self.assertTrue(
                recorder.record_mllm_event(
                    {
                        "stage": "M1",
                        "timestamp": time.time(),
                        "m1_input_image_data_url": image_data,
                        "raw_text": '{"state":"closed"}',
                    }
                )
            )
            self.assertTrue(recorder.record_panel(1, bytes(packet["rgb"]["data"], "ascii")))
            self.assertTrue(recorder.record_phone_frame(bytes(packet["rgb"]["data"], "ascii"), sequence=1))
            self.assertTrue(recorder.record_phone_audio(b"\x00\x00" * 100, sample_rate=48_000))
            self.assertTrue(recorder.record_state_snapshot({"detections": [{"bbox": [1, 2, 3, 4], "mask": {"rows": [1]}}, {"m1_input_image_data_url": "data:image/jpeg;base64,AAAA"}]}))
            self.assertTrue(recorder.record_step_boundary(step_index=0, stamp=1.2, frame_seq=3))
            stopped = recorder.stop()
            self.assertFalse(stopped["active"])
            session = pathlib.Path(stopped["record_dir"])
            self.assertTrue((session / "raw/camera/manifest.jsonl").is_file())
            camera_record = json.loads(
                (session / "raw/camera/manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(camera_record["stamp"], 1.2)
            self.assertEqual(
                camera_record["capture_timing"], packet["capture_timing"]
            )
            self.assertIn("session_time_s", camera_record)
            for stream in ("rgb", "depth"):
                self.assertEqual(
                    (session / camera_record["files"][stream]).read_bytes(),
                    base64.b64decode(packet[stream]["data"]),
                )
            self.assertTrue((session / "raw/map_manifest.jsonl").is_file())
            self.assertTrue((session / "raw/panels/manifest.jsonl").is_file())
            self.assertTrue((session / "raw/phone/audio.wav").is_file())
            self.assertTrue((session / "raw/phone/manifest.jsonl").is_file())
            state_line = (session / "raw/state/right_panel.jsonl").read_text(encoding="utf-8")
            self.assertNotIn('"mask"', state_line)
            self.assertNotIn("m1_input_image_data_url", state_line)
            map_line = (session / "raw/map_manifest.jsonl").read_text(encoding="utf-8")
            self.assertNotIn('"data"', map_line)
            mllm_line = (session / "raw/semantic/mllm_events.jsonl").read_text(encoding="utf-8")
            self.assertIn("m1_input_image_path", mllm_line)
            self.assertEqual(json.loads((session / "session.json").read_text())["status"], "complete")

    def test_page_capture_skips_high_volume_rgbd_and_map_rasters(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = PhysicalRawRecorder(temporary, default_mode="page_capture")
            recorder.start(mode="page_capture")
            self.assertFalse(recorder.record_sensor_packet(self._packet()))
            self.assertFalse(
                recorder.record_ros_state(
                    "occupancy", {"width": 1, "height": 1, "resolution": 0.1, "data": [0]}
                )
            )
            result = recorder.stop()
            session = pathlib.Path(result["record_dir"])
            self.assertFalse((session / "raw/camera/manifest.jsonl").exists())
            self.assertFalse((session / "raw/map_manifest.jsonl").exists())

    def test_dark_showcase_summary_contains_m3(self):
        self.assertIn("state?.mllm?.M3", DARK_SHOWCASE_HTML)
        self.assertIn("M1 / M2 / M3", DARK_SHOWCASE_HTML)
        self.assertIn("验证问题", DARK_SHOWCASE_HTML)
        self.assertIn("/api/recording/start", DARK_SHOWCASE_HTML)
        self.assertIn("后台录制到本机", DARK_SHOWCASE_HTML)

    def test_offline_compositor_writes_theme_and_right_rail_outputs(self):
        import subprocess

        with tempfile.TemporaryDirectory() as temporary:
            recorder = PhysicalRawRecorder(temporary)
            recorder.start(label="offline-test")
            packet = self._packet()
            self.assertTrue(recorder.record_sensor_packet(packet))
            self.assertTrue(recorder.record_panel(1, b"not-a-jpeg"))
            self.assertTrue(recorder.record_phone_frame(bytes(packet["rgb"]["data"], "ascii"), sequence=1))
            self.assertTrue(recorder.record_state_snapshot({"telemetry": {}, "navigation": {}, "graph": {}}))
            self.assertTrue(recorder.record_step_boundary(step_index=0, stamp=0.0, frame_seq=3))
            result = recorder.stop()
            session = pathlib.Path(result["record_dir"])
            command = [
                sys.executable,
                str(ROOT.parents[0] / "build_physical_showcase_video.py"),
                str(session),
                "--theme",
                "academic",
                "--max-frames",
                "1",
                "--phone-mode",
                "pip",
            ]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("physical_showcase_derived_v1", completed.stdout)
            derived = session / "derived" / "showcase-academic"
            self.assertTrue((derived / "showcase.mp4").is_file())
            self.assertTrue((derived / "right_rail.mp4").is_file())
            self.assertTrue((derived / "right_rail.jsonl").is_file())

    def test_offline_snapshot_deduplicates_state_and_event_stream_mllm_rows(self):
        from build_physical_showcase_video import SessionData

        with tempfile.TemporaryDirectory() as temporary:
            recorder = PhysicalRawRecorder(temporary)
            recorder.start()
            event = {"event_id": "same-call", "stage": "M3", "timestamp": time.time(), "result": {"state": "open"}}
            self.assertTrue(recorder.record_mllm_event(event))
            self.assertTrue(recorder.record_state_snapshot({"mllm_events": [event], "navigation": {}}))
            self.assertTrue(recorder.record_step_boundary(step_index=0, stamp=0.0))
            session = pathlib.Path(recorder.stop()["record_dir"])
            snapshot = SessionData(session).snapshot(10.0)
            self.assertEqual(len(snapshot.get("mllm_events", [])), 1)


if __name__ == "__main__":
    unittest.main()
