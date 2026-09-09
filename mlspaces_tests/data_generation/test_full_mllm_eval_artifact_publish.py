"""Offline tests for shallow full-MLLM episode artifact publication."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "InteractiveNav"
    / "run_full_mllm_interactive_nav_eval.py"
)
SPEC = importlib.util.spec_from_file_location("full_mllm_interactive_nav_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_publish_task_artifacts_keeps_sources_and_exposes_shallow_files(tmp_path: Path) -> None:
    task_dir = tmp_path / "mixed" / "episode_0001_case"
    episode_dir = task_dir / "attempt_001" / "eval" / "episodes" / "0001_case"
    video_dir = task_dir / "attempt_001" / "videos"
    episode_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    source_topdown = episode_dir / "episode_topdown.png"
    source_metadata = episode_dir / "episode_topdown.json"
    source_video = video_dir / "overview_6panel.mp4"
    source_topdown.write_bytes(b"png-data")
    source_metadata.write_text('{"coverage": 0.5}\n', encoding="utf-8")
    source_video.write_bytes(b"mp4-data")

    published = MODULE.publish_task_artifacts(
        task_dir,
        {
            "topdown_path": str(source_topdown),
            "topdown_metadata_path": str(source_metadata),
            "six_panel_video": str(source_video),
        },
        recording_required=True,
        topdown_required=True,
    )

    assert source_topdown.read_bytes() == b"png-data"
    assert source_metadata.is_file()
    assert source_video.read_bytes() == b"mp4-data"
    assert (task_dir / "episode_topdown.png").read_bytes() == b"png-data"
    assert (task_dir / "episode_topdown.json").read_text(encoding="utf-8") == '{"coverage": 0.5}\n'
    assert (task_dir / "overview_6panel.mp4").read_bytes() == b"mp4-data"
    assert published["source_topdown_path"] == str(source_topdown)
    assert published["source_six_panel_video"] == str(source_video)
    assert published["topdown_path"] == str(task_dir / "episode_topdown.png")
    assert published["six_panel_video"] == str(task_dir / "overview_6panel.mp4")
    assert published["artifact_publish_valid"] is True

    # Resume/reconciliation may publish the same row again; this must be safe.
    repeated = MODULE.publish_task_artifacts(
        task_dir,
        published,
        recording_required=True,
        topdown_required=True,
    )
    assert repeated["artifact_publish_valid"] is True


def test_publish_task_artifacts_allows_disabled_optional_outputs(tmp_path: Path) -> None:
    task_dir = tmp_path / "channel" / "episode_0000_case"
    published = MODULE.publish_task_artifacts(
        task_dir,
        {
            "topdown_path": str(task_dir / "attempt_001" / "missing_topdown.png"),
            "topdown_metadata_path": str(task_dir / "attempt_001" / "missing_topdown.json"),
            "six_panel_video": str(task_dir / "attempt_001" / "missing_video.mp4"),
        },
        recording_required=False,
        topdown_required=False,
    )

    assert published["topdown_exists"] is False
    assert published["recording_artifact_valid"] is True
    assert published["topdown_artifact_valid"] is True
    assert published["artifact_publish_valid"] is True


def test_resume_backfills_shallow_artifacts(tmp_path: Path) -> None:
    task_dir = tmp_path / "container" / "episode_0002_case"
    episode_dir = task_dir / "attempt_001" / "eval" / "episodes" / "0002_case"
    video_dir = task_dir / "attempt_001" / "videos"
    episode_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    source_topdown = episode_dir / "episode_topdown.png"
    source_metadata = episode_dir / "episode_topdown.json"
    source_video = video_dir / "overview_6panel.mp4"
    source_topdown.write_bytes(b"png")
    source_metadata.write_text("{}\n", encoding="utf-8")
    source_video.write_bytes(b"video")
    summary_path = task_dir / "batch_task_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "result_complete": True,
                "recording_enabled": True,
                "topdown_path": str(source_topdown),
                "topdown_metadata_path": str(source_metadata),
                "six_panel_video": str(source_video),
            }
        ),
        encoding="utf-8",
    )
    task = MODULE.EpisodeTask(
        ordinal=2,
        domain="container",
        local_index=2,
        case_id="case",
        house_index=3,
        benchmark="benchmark.json",
        worker_id=0,
        master_port=13500,
        task_dir=str(task_dir),
    )

    resumed = MODULE.run_task(
        task,
        SimpleNamespace(resume=True, recording=True, render_topdown=True),
    )

    assert resumed["resumed"] is True
    assert resumed["topdown_path"] == str(task_dir / "episode_topdown.png")
    assert resumed["six_panel_video"] == str(task_dir / "overview_6panel.mp4")
    assert json.loads(summary_path.read_text(encoding="utf-8"))["artifact_publish_valid"] is True

    # A missing shallow alias must be repaired from the retained attempt source,
    # not interpreted as a reason to launch a new attempt.
    (task_dir / "episode_topdown.png").unlink()
    (task_dir / "episode_topdown.json").unlink()
    (task_dir / "overview_6panel.mp4").unlink()
    repaired = MODULE.run_task(
        task,
        SimpleNamespace(resume=True, recording=True, render_topdown=True),
    )
    assert repaired["resumed"] is True
    assert (task_dir / "episode_topdown.png").is_file()
    assert (task_dir / "overview_6panel.mp4").is_file()


def test_publish_batch_galleries_uses_copies_relative_links_and_root_sheet(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "batch"
    tasks = [
        MODULE.EpisodeTask(
            ordinal=0,
            domain="channel",
            local_index=0,
            case_id="channel_case",
            house_index=2,
            benchmark="channel.json",
            worker_id=0,
            master_port=13500,
            task_dir=str(output_dir / "channel" / "episode_0000_channel_case"),
        ),
        MODULE.EpisodeTask(
            ordinal=1,
            domain="mixed",
            local_index=1,
            case_id="mixed_case",
            house_index=153,
            benchmark="mixed.json",
            worker_id=1,
            master_port=13501,
            task_dir=str(output_dir / "mixed" / "episode_0001_mixed_case"),
        ),
        # A failed/partial episode without artifacts must not block valid rows.
        MODULE.EpisodeTask(
            ordinal=2,
            domain="container",
            local_index=2,
            case_id="missing_case",
            house_index=9,
            benchmark="container.json",
            worker_id=0,
            master_port=13500,
            task_dir=str(output_dir / "container" / "episode_0002_missing_case"),
        ),
    ]
    rows = []
    source_paths = []
    for task, color in zip(tasks[:2], ((20, 120, 30), (30, 40, 160)), strict=True):
        attempt_dir = Path(task.task_dir) / "attempt_001"
        episode_dir = attempt_dir / "eval" / "episodes" / task.case_id
        episode_dir.mkdir(parents=True)
        video = attempt_dir / "videos" / "overview_6panel.mp4"
        video.parent.mkdir(parents=True)
        topdown = episode_dir / "episode_topdown.png"
        metadata = episode_dir / "episode_topdown.json"
        Image.new("RGB", (120, 80), color).save(topdown)
        metadata.write_text(
            json.dumps(
                {
                    "coverage": {
                        "exploration_coverage_ratio": 0.5,
                        "mapped_free_coverage_ratio": 0.4,
                    },
                    "scene_background": {"map_source": "test", "mode": "full"},
                }
            ),
            encoding="utf-8",
        )
        video.write_bytes(b"not-a-real-mp4-but-a-valid-nonempty-artifact")
        source_paths.append((topdown, metadata, video))
        rows.append(
            {
                "domain": task.domain,
                "local_index": task.local_index,
                "source_topdown_path": str(topdown),
                "source_topdown_metadata_path": str(metadata),
                "source_six_panel_video": str(video),
            }
        )

    summary = MODULE.publish_batch_galleries(
        output_dir,
        tasks,
        rows,
        recording_enabled=True,
        topdown_enabled=True,
    )

    assert summary["topdown_count"] == 2
    assert summary["video_count"] == 2
    assert summary["error_count"] == 0
    assert (output_dir / "contact_sheet_all.png").is_file()
    assert not (output_dir / "topdown_gallery" / "contact_sheet_all.png").exists()
    for task, (source_topdown, source_metadata, source_video) in zip(
        tasks[:2], source_paths, strict=True
    ):
        stem = MODULE._gallery_stem(task)
        copied_topdown = output_dir / "topdown_gallery" / f"{stem}_topdown.png"
        copied_metadata = output_dir / "topdown_gallery" / f"{stem}_topdown.json"
        linked_video = output_dir / "video_gallery" / f"{stem}_overview_6panel.mp4"
        assert copied_topdown.read_bytes() == source_topdown.read_bytes()
        assert copied_metadata.read_bytes() == source_metadata.read_bytes()
        assert not copied_topdown.is_symlink()
        assert linked_video.is_symlink()
        assert not Path(os.readlink(linked_video)).is_absolute()
        assert linked_video.resolve() == source_video.resolve()

    topdown_index = MODULE.read_json(output_dir / "topdown_gallery" / "index.json")
    video_index = MODULE.read_json(output_dir / "video_gallery" / "index.json")
    assert topdown_index["count"] == 2
    assert topdown_index["contact_sheet"] == "../contact_sheet_all.png"
    assert video_index["count"] == 2
    assert video_index["symlink_count"] == 2

    # Resume/final reconciliation repairs stale destination types and targets.
    first_stem = MODULE._gallery_stem(tasks[0])
    stale_topdown = output_dir / "topdown_gallery" / f"{first_stem}_topdown.png"
    stale_video = output_dir / "video_gallery" / f"{first_stem}_overview_6panel.mp4"
    stale_topdown.unlink()
    stale_topdown.symlink_to(source_paths[0][0])
    stale_video.unlink()
    stale_video.symlink_to("missing.mp4")
    repeated = MODULE.publish_batch_galleries(
        output_dir,
        tasks,
        rows,
        recording_enabled=True,
        topdown_enabled=True,
    )
    assert repeated["error_count"] == 0
    assert not stale_topdown.is_symlink()
    assert stale_video.is_symlink()
    assert stale_video.resolve() == source_paths[0][2].resolve()


def test_publish_batch_galleries_respects_disabled_artifact_switches(tmp_path: Path) -> None:
    output_dir = tmp_path / "disabled"
    summary = MODULE.publish_batch_galleries(
        output_dir,
        [],
        [],
        recording_enabled=False,
        topdown_enabled=False,
    )

    assert summary["topdown_gallery"] is None
    assert summary["video_gallery"] is None
    assert not (output_dir / "topdown_gallery").exists()
    assert not (output_dir / "video_gallery").exists()
    assert not (output_dir / "contact_sheet_all.png").exists()


def test_publish_batch_galleries_reports_but_tolerates_completed_missing_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "missing"
    task = MODULE.EpisodeTask(
        ordinal=0,
        domain="container",
        local_index=0,
        case_id="missing",
        house_index=7,
        benchmark="container.json",
        worker_id=0,
        master_port=13500,
        task_dir=str(output_dir / "container" / "missing"),
    )

    summary = MODULE.publish_batch_galleries(
        output_dir,
        [task],
        [{"domain": "container", "local_index": 0, "result_complete": True}],
        recording_enabled=True,
        topdown_enabled=True,
    )

    assert summary["topdown_count"] == 0
    assert summary["video_count"] == 0
    assert summary["error_count"] == 3
    assert any("missing PNG" in error for error in summary["errors"])
    assert any("missing JSON sidecar" in error for error in summary["errors"])
    assert any("missing overview_6panel.mp4" in error for error in summary["errors"])
    assert MODULE.read_json(output_dir / "topdown_gallery" / "index.json")["count"] == 0
    assert MODULE.read_json(output_dir / "video_gallery" / "index.json")["count"] == 0
    assert not (output_dir / "contact_sheet_all.png").exists()


def test_topdown_failure_metric_is_disabled_with_topdown_rendering(tmp_path: Path) -> None:
    task = MODULE.EpisodeTask(
        ordinal=0,
        domain="mixed",
        local_index=0,
        case_id="case",
        house_index=1,
        benchmark="mixed.json",
        worker_id=0,
        master_port=13500,
        task_dir=str(tmp_path / "mixed" / "case"),
    )
    row = {
        "domain": "mixed",
        "local_index": 0,
        "result_complete": True,
        "success": False,
        "task_success": False,
        "nav_success": False,
        "topdown_exists": False,
    }

    disabled = MODULE.aggregate_results(
        [task], [row], SimpleNamespace(recording=False, render_topdown=False)
    )
    enabled = MODULE.aggregate_results(
        [task], [row], SimpleNamespace(recording=False, render_topdown=True)
    )

    assert disabled["topdown_failure_count"] == 0
    assert disabled["domains"]["mixed"]["topdown_failure_count"] == 0
    assert enabled["topdown_failure_count"] == 1
    assert enabled["domains"]["mixed"]["topdown_failure_count"] == 1
