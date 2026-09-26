import importlib.util
import io
import json
import queue
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from physical_raw_recorder import PhysicalRawRecorder
from physical_protocol import image_packet
import recording_control


def test_full_recording_queue_never_waits_on_critical_event(tmp_path):
    recorder = PhysicalRawRecorder(tmp_path)
    recorder._accepting = True
    recorder._queue = queue.Queue(maxsize=1)
    recorder._queue.put_nowait(object())
    started = time.monotonic()
    assert not recorder._enqueue("event", "test", {}, critical=True)
    assert time.monotonic() - started < .1
    assert recorder._degraded
    assert recorder._stats["queue_dropped"]["test"] == 1


def test_homepage_record_buttons_use_backend_only():
    from physical_six_panel_server import _HTML
    assert "id='record-start'" in _HTML
    assert "id='record-stop'" in _HTML
    assert "mode:'raw_plus_panels'" in _HTML
    assert "getDisplayMedia" not in _HTML


def exporter():
    path = Path(__file__).resolve().parents[2] / "build_physical_six_panel_video.py"
    spec = importlib.util.spec_from_file_location("six_panel_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_record_and_export_exact_overview_and_raw_camera(tmp_path):
    recorder = PhysicalRawRecorder(tmp_path / "recordings")
    started = recorder.start()
    camera = np.full((24, 32, 3), (10, 80, 180), dtype=np.uint8)
    overview = np.full((48, 96, 3), (100, 70, 30), dtype=np.uint8)
    rgb = cv2.imencode(".jpg", camera)[1].tobytes()
    depth = cv2.imencode(".png", np.ones((24, 32), np.uint16))[1].tobytes()
    panel = cv2.imencode(".jpg", overview)[1].tobytes()
    packet = image_packet(seq=1, stamp=1.0, rgb_jpeg=rgb, depth_png=depth,
                          width=32, height=24, camera_frame="camera", depth_scale=.001,
                          intrinsics={"fx": 20, "fy": 20, "cx": 16, "cy": 12},
                          color_depth_sync_ms=0)
    assert recorder.record_sensor_packet(packet)
    assert recorder.record_panel(0, panel, stamp=1, frame_seq=1)
    assert recorder.record_panel(5, panel, stamp=1, frame_seq=1)
    recorder.stop()
    session = Path(started["record_dir"])
    assert (session / "raw/camera/first_person.mjpeg").read_bytes() == rgb
    records = [json.loads(line) for line in (session / "raw/panels/manifest.jsonl").read_text().splitlines()]
    assert {row["panel_index"] for row in records} == {0, 5}
    assert (session / records[0]["image"]).read_bytes() == panel
    module = exporter()
    result = module.build(session, tmp_path / "export", 10)
    assert len(result["videos"]) == 2
    for video in result["videos"]:
        cap = cv2.VideoCapture(str(tmp_path / "export" / video["file"]))
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == video["width"]
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == video["frames"]
        cap.release()
    with pytest.raises(FileExistsError):
        module.build(session, tmp_path / "export", 10)


def test_no_future_frame_is_shown():
    module = exporter()
    assert module.frame_at([1, 2, 4], .5) == -1
    assert module.frame_at([1, 2, 4], 2) == 1
    assert module.frame_at([1, 2, 4], 3) == 1


def test_step_overlay_adds_header_without_cropping_source(tmp_path):
    module = exporter()
    source = np.full((48, 640, 3), 100, np.uint8)
    cv2.imwrite(str(tmp_path / "source.jpg"), source)
    output = tmp_path / "steps.mp4"
    result = module.export_stream(tmp_path, [{"session_time_s": 0, "image": "source.jpg",
                                            "step_index": 428}], output, 10, .2, step_overlay=True)
    assert result["height"] == 108
    cap = cv2.VideoCapture(str(output))
    ok, image = cap.read()
    cap.release()
    assert ok and image.shape[:2] == (108, 640)
    assert image[:60].max() > 200
    assert abs(float(image[60:].mean()) - 100) < 10


def test_step_overlay_keeps_repeated_camera_frame_steps(tmp_path, monkeypatch):
    module = exporter()
    session = tmp_path / "session"
    (session / "raw/panels").mkdir(parents=True)
    (session / "raw/camera").mkdir()
    (session / "session.json").write_text(json.dumps({"status": "complete", "duration_s": 1}))
    panels = [{"panel_index": 0, "frame_seq": 10, "session_time_s": i * .4,
               "image": f"{i}.jpg"} for i in range(2)]
    boundaries = [{"frame_seq": 10, "step_index": 428 + i} for i in range(2)]
    (session / "raw/panels/manifest.jsonl").write_text("\n".join(map(json.dumps, panels)))
    (session / "raw/step_boundaries.jsonl").write_text("\n".join(map(json.dumps, boundaries)))
    (session / "raw/camera/manifest.jsonl").write_text(json.dumps({"session_time_s": 0, "files": {"rgb": "rgb.jpg"}}))
    captured = []
    def export(_session, records, *args, **kwargs):
        captured.extend(r["step_index"] for r in records)
        return {}
    monkeypatch.setattr(module, "export_stream", export)
    module.build(session, tmp_path / "output", step_overlay=True, overview_only=True)
    assert captured == [428, 429]


def test_record_option_is_parsed_before_legacy_motion_arguments():
    source = (Path(__file__).resolve().parents[1] / "physical_nav_all.sh").read_text()
    assert source.index('== "--record"') < source.index('if [[ "${2:-}" == "enable_motion"')
    assert "export PHYSICAL_NAV_RECORD_MODE=raw_plus_panels" in source
    assert '"${ROOT_DIR}/recording_control.py" start' in source
    assert '"${ROOT_DIR}/recording_control.py" stop' in source


@pytest.mark.parametrize("action", ["start", "stop"])
def test_launcher_recording_lifecycle(monkeypatch, action):
    requests = []
    def open_request(request, timeout):
        requests.append(request)
        payload = ({"active": True, "session_id": "test", "mode": "raw_plus_panels"}
                   if request.full_url.endswith("status") or action == "start"
                   else {"active": False})
        return io.BytesIO(json.dumps(payload).encode())
    class Opener:
        open = staticmethod(open_request)
    monkeypatch.setattr(recording_control.urllib.request, "build_opener", lambda *_: Opener())
    recording_control.control(action, 8765)
    assert len(requests) == 2
    assert requests[-1].full_url.endswith("/" + action)
