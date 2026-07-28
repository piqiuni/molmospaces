from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERACTIVE_NAV_SCRIPTS = REPO_ROOT / "scripts" / "InteractiveNav"
if str(INTERACTIVE_NAV_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(INTERACTIVE_NAV_SCRIPTS))

from wait_for_recorder_drain import (
    count_csv_frames,
    expected_frames_from_episode_result,
    expected_video_frames_from_episode_result,
    expected_video_frames_from_step_count,
    final_recorder_summary_errors,
    wait_for_recorder_frames,
)


def _write_frames(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "step_id"])
        for index in range(count):
            writer.writerow([index + 1, index])


def test_expected_frames_and_csv_count(tmp_path: Path) -> None:
    result_path = tmp_path / "episode_result.json"
    result_path.write_text(
        json.dumps({"status": "complete", "result": {"step_count": 3}}),
        encoding="utf-8",
    )
    frames_path = tmp_path / "video_frames.csv"
    _write_frames(frames_path, 2)

    assert expected_frames_from_episode_result(result_path) == 3
    assert expected_video_frames_from_episode_result(result_path, 2) == 2
    assert count_csv_frames(frames_path) == 2
    assert count_csv_frames(tmp_path / "missing.csv") == 0


def test_expected_frames_accepts_legacy_top_level_step_count(tmp_path: Path) -> None:
    result_path = tmp_path / "episode_result.json"
    result_path.write_text(json.dumps({"step_count": 5}), encoding="utf-8")

    assert expected_frames_from_episode_result(result_path) == 5
    assert expected_video_frames_from_step_count(5, 2) == 3


def test_sampled_video_frame_count_rejects_invalid_interval() -> None:
    try:
        expected_video_frames_from_step_count(5, 0)
    except ValueError as exc:
        assert "step_sync_capture_every" in str(exc)
    else:
        raise AssertionError("zero sampling interval must be rejected")


def test_wait_for_recorder_frames_tracks_expected_evaluator_steps(tmp_path: Path) -> None:
    frames_path = tmp_path / "video_frames.csv"
    _write_frames(frames_path, 0)
    clock = [0.0]

    def monotonic() -> float:
        return clock[0]

    def sleep(_seconds: float) -> None:
        clock[0] += 0.5
        _write_frames(frames_path, min(3, int(clock[0] * 2)))

    result = wait_for_recorder_frames(
        frames_path,
        3,
        timeout_seconds=5.0,
        poll_seconds=0.5,
        monotonic=monotonic,
        sleep=sleep,
    )

    assert result.complete is True
    assert result.recorded_frames == 3
    assert result.elapsed_seconds == 1.5


def test_wait_for_recorder_frames_has_bounded_timeout(tmp_path: Path) -> None:
    frames_path = tmp_path / "video_frames.csv"
    _write_frames(frames_path, 1)
    clock = [0.0]

    def monotonic() -> float:
        return clock[0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    result = wait_for_recorder_frames(
        frames_path,
        4,
        timeout_seconds=1.0,
        poll_seconds=0.25,
        monotonic=monotonic,
        sleep=sleep,
    )

    assert result.complete is False
    assert result.recorded_frames == 1
    assert result.elapsed_seconds == 1.0


def test_final_summary_requires_nondropping_runtime_video(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "finalization_complete": True,
                "step_sync_count": 4,
                "first_person_video_frame_count": 4,
                "step_sync_placeholder_count": 0,
                "video_frame_jobs_dropped": 0,
                "runtime_video_encode": True,
                "artifact_writer_stats": {"written_video_jobs": 4},
                "first_person_video_error": "",
            }
        ),
        encoding="utf-8",
    )
    assert final_recorder_summary_errors(summary_path, 4) == []

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["artifact_writer_stats"]["written_video_jobs"] = 3
    payload["step_sync_placeholder_count"] = 1
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    errors = final_recorder_summary_errors(summary_path, 4)
    assert "step_sync_placeholder_count=1" in errors
    assert "written_video_jobs=3 < expected=4" in errors


def test_final_summary_validates_raw_steps_and_sampled_video_separately(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "finalization_complete": True,
                "step_sync_count": 4,
                "first_person_video_frame_count": 2,
                "step_sync_placeholder_count": 0,
                "video_frame_jobs_dropped": 0,
                "runtime_video_encode": True,
                "artifact_writer_stats": {"written_video_jobs": 2},
                "first_person_video_error": "",
            }
        ),
        encoding="utf-8",
    )
    assert final_recorder_summary_errors(summary_path, 4, 2) == []

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["step_sync_count"] = 3
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    assert "step_sync_count=3 < expected=4" in final_recorder_summary_errors(
        summary_path, 4, 2
    )
