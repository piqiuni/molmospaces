#!/usr/bin/env python3
"""Wait until the ROS debug recorder has rendered every evaluator step."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class DrainResult:
    expected_frames: int
    recorded_frames: int
    elapsed_seconds: float
    complete: bool
    recorder_alive: bool | None


def expected_frames_from_episode_result(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("result") or {}
    value = int(payload.get("step_count", result.get("step_count", -1)))
    if value < 0:
        raise ValueError(f"episode result has invalid step_count={value}: {path}")
    return value


def expected_video_frames_from_step_count(
    expected_steps: int, step_sync_capture_every: int,
) -> int:
    if expected_steps < 0:
        raise ValueError(f"expected_steps must be non-negative, got {expected_steps}")
    if step_sync_capture_every < 1:
        raise ValueError(
            "step_sync_capture_every must be at least one, got "
            f"{step_sync_capture_every}"
        )
    return (expected_steps + step_sync_capture_every - 1) // step_sync_capture_every


def expected_video_frames_from_episode_result(
    path: Path, step_sync_capture_every: int = 1,
) -> int:
    return expected_video_frames_from_step_count(
        expected_frames_from_episode_result(path), step_sync_capture_every
    )


def final_recorder_summary_errors(
    path: Path,
    expected_steps: int,
    expected_video_frames: int | None = None,
) -> list[str]:
    if expected_video_frames is None:
        expected_video_frames = expected_steps
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"missing recorder summary: {path}"]
    errors = []
    if payload.get("finalization_complete") is not True:
        errors.append("recorder finalization is incomplete")
    step_sync_count = int(payload.get("step_sync_count", -1))
    if step_sync_count < expected_steps:
        errors.append(f"step_sync_count={step_sync_count} < expected={expected_steps}")
    video_frame_count = int(payload.get("first_person_video_frame_count", -1))
    if video_frame_count < expected_video_frames:
        errors.append(
            "first_person_video_frame_count="
            f"{video_frame_count} < expected={expected_video_frames}"
        )
    placeholder_count = int(payload.get("step_sync_placeholder_count", 0) or 0)
    if placeholder_count:
        errors.append(f"step_sync_placeholder_count={placeholder_count}")
    dropped_frames = int(payload.get("video_frame_jobs_dropped", 0) or 0)
    has_drop_breakdown = (
        "video_frame_jobs_dropped_oldest" in payload
        or "video_frame_jobs_dropped_newest" in payload
    )
    dropped_newest = int(payload.get("video_frame_jobs_dropped_newest", 0) or 0)
    if has_drop_breakdown:
        # ``drop_oldest`` is an intentional freshness policy.  The strict
        # frame-count check above still catches an incomplete video, but do
        # not mislabel an explicitly recorded oldest-job eviction as an RGB
        # placeholder or an unexplained newest-frame loss.
        if dropped_newest:
            errors.append(
                "video_frame_jobs_dropped_newest="
                f"{dropped_newest} (total={dropped_frames})"
            )
    elif dropped_frames:
        # Summaries written before the split counters remain strict.
        errors.append(f"video_frame_jobs_dropped={dropped_frames}")
    writer_stats = payload.get("artifact_writer_stats") or {}
    written_video_jobs = int(writer_stats.get("written_video_jobs", -1))
    if (
        bool(payload.get("runtime_video_encode"))
        and written_video_jobs < expected_video_frames
    ):
        errors.append(
            "written_video_jobs="
            f"{written_video_jobs} < expected={expected_video_frames}"
        )
    video_error = str(payload.get("first_person_video_error") or "").strip()
    if video_error:
        errors.append(f"first_person_video_error={video_error}")
    return errors


def count_csv_frames(path: Path) -> int:
    """Count complete data records without treating a partial final line as a frame."""

    try:
        with path.open("rb") as handle:
            complete_lines = sum(
                bool(line.strip()) and line.endswith(b"\n") for line in handle
            )
            return max(0, complete_lines - 1)
    except FileNotFoundError:
        return 0


def process_is_alive(pid: int | None) -> bool | None:
    if pid is None:
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_recorder_frames(
    csv_path: Path,
    expected_frames: int,
    *,
    timeout_seconds: float,
    poll_seconds: float = 0.5,
    recorder_pid: int | None = None,
    progress_seconds: float = 10.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[DrainResult], None] | None = None,
) -> DrainResult:
    if expected_frames < 0:
        raise ValueError("expected_frames must be non-negative")
    timeout_seconds = max(0.0, float(timeout_seconds))
    poll_seconds = max(0.01, float(poll_seconds))
    progress_seconds = max(poll_seconds, float(progress_seconds))
    started = monotonic()
    deadline = started + timeout_seconds
    next_progress = started

    while True:
        now = monotonic()
        recorded = max(0, count_csv_frames(csv_path))
        alive = process_is_alive(recorder_pid)
        result = DrainResult(
            expected_frames=expected_frames,
            recorded_frames=recorded,
            elapsed_seconds=max(0.0, now - started),
            complete=recorded >= expected_frames,
            recorder_alive=alive,
        )
        if result.complete:
            if progress is not None:
                progress(result)
            return result
        if alive is False or now >= deadline:
            if progress is not None:
                progress(result)
            return result
        if progress is not None and now >= next_progress:
            progress(result)
            next_progress = now + progress_seconds
        sleep(min(poll_seconds, max(0.0, deadline - now)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-result", type=Path, required=True)
    parser.add_argument("--video-frames-csv", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--poll-sec", type=float, default=0.5)
    parser.add_argument("--progress-sec", type=float, default=10.0)
    parser.add_argument("--recorder-pid", type=int)
    parser.add_argument(
        "--step-sync-capture-every",
        type=int,
        default=1,
        help="Expected six-panel sampling interval; raw step-sync markers are still all required.",
    )
    parser.add_argument(
        "--recorder-summary",
        type=Path,
        help="After shutdown, additionally require a finalized, non-dropped runtime video.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    expected_steps = expected_frames_from_episode_result(args.episode_result)
    expected = expected_video_frames_from_step_count(
        expected_steps, args.step_sync_capture_every
    )

    def report(result: DrainResult) -> None:
        status = "complete" if result.complete else "waiting"
        if result.recorder_alive is False:
            status = "recorder_exited"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S%z")
        print(
            "[recorder-drain] "
            f"time={timestamp} status={status} "
            f"frames={result.recorded_frames}/{result.expected_frames} "
            f"elapsed={result.elapsed_seconds:.1f}s",
            flush=True,
        )

    result = wait_for_recorder_frames(
        args.video_frames_csv,
        expected,
        timeout_seconds=args.timeout_sec,
        poll_seconds=args.poll_sec,
        recorder_pid=args.recorder_pid,
        progress_seconds=args.progress_sec,
        progress=report,
    )
    if not result.complete:
        return 1
    if args.recorder_summary is not None:
        errors = final_recorder_summary_errors(
            args.recorder_summary,
            expected_steps,
            expected,
        )
        if errors:
            for error in errors:
                print(f"[recorder-drain] final_validation_error={error}", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
