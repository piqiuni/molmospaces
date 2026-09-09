import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).with_name("run_full_mllm_interactive_nav_eval.py")
SPEC = importlib.util.spec_from_file_location(
    "run_full_mllm_interactive_nav_eval",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
batch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batch
SPEC.loader.exec_module(batch)


def _task(tmp_path: Path) -> object:
    return batch.EpisodeTask(
        ordinal=0,
        domain="mixed",
        local_index=1,
        case_id="mixed_case",
        house_index=153,
        benchmark=str(tmp_path / "mixed.json"),
        worker_id=0,
        master_port=14000,
        task_dir=str(tmp_path / "mixed" / "episode_0001_mixed_case"),
    )


def _write_attempt_artifacts(attempt_dir: Path) -> dict[str, str]:
    episode_dir = attempt_dir / "eval" / "episodes" / "0001_mixed_case"
    episode_dir.mkdir(parents=True, exist_ok=True)
    topdown = episode_dir / "episode_topdown.png"
    metadata = episode_dir / "episode_topdown.json"
    video = attempt_dir / "videos" / "overview_6panel.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    topdown.write_bytes(b"topdown-image")
    metadata.write_text(json.dumps({"coverage": 0.42}), encoding="utf-8")
    video.write_bytes(b"six-panel-video")
    return {
        "topdown_path": str(topdown),
        "topdown_metadata_path": str(metadata),
        "six_panel_video": str(video),
    }


def _assert_published(task_dir: Path, source_paths: dict[str, str], row: dict) -> None:
    published = {
        "topdown_path": task_dir / "episode_topdown.png",
        "topdown_metadata_path": task_dir / "episode_topdown.json",
        "six_panel_video": task_dir / "overview_6panel.mp4",
    }
    source_keys = {
        "topdown_path": "source_topdown_path",
        "topdown_metadata_path": "source_topdown_metadata_path",
        "six_panel_video": "source_six_panel_video",
    }
    for active_key, destination in published.items():
        source = Path(source_paths[active_key])
        assert source.is_file()
        assert destination.is_file()
        assert destination.read_bytes() == source.read_bytes()
        assert row[active_key] == str(destination)
        assert row[source_keys[active_key]] == str(source)
    assert row["artifact_publish_valid"] is True
    assert row["artifact_publish_errors"] == []


def test_run_task_publishes_normal_completed_attempt(tmp_path, monkeypatch):
    task = _task(tmp_path)
    source_paths: dict[str, str] = {}

    class FakeProcess:
        pid = 12345

        def wait(self, timeout):
            return 0

    def fake_parse_episode_result(attempt_task, _args):
        source_paths.update(_write_attempt_artifacts(Path(attempt_task.task_dir)))
        return {
            **source_paths,
            "result_complete": True,
            "success": False,
            "task_success": False,
            "nav_success": False,
        }

    monkeypatch.setattr(batch, "make_environment", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(batch, "parse_episode_result", fake_parse_episode_result)
    args = SimpleNamespace(
        resume=False,
        recording=True,
        render_topdown=True,
        dry_run=False,
        runner=SCRIPT_PATH,
        runner_mode="v3",
        runner_shell="bash",
        scene_timeout_s=10.0,
        model_endpoints=None,
    )

    row = batch.run_task(task, args)

    _assert_published(Path(task.task_dir), source_paths, row)
    persisted = batch.read_json(Path(task.task_dir) / "batch_task_summary.json")
    _assert_published(Path(task.task_dir), source_paths, persisted)


def test_run_task_resume_republishes_from_deep_sources(tmp_path):
    task = _task(tmp_path)
    task_dir = Path(task.task_dir)
    source_paths = _write_attempt_artifacts(task_dir / "attempt_001")
    # A retry/resume may find stale shallow aliases from an earlier attempt;
    # publishing must replace them without modifying the authoritative source.
    (task_dir / "episode_topdown.png").write_bytes(b"stale-image")
    (task_dir / "episode_topdown.json").write_text("{}", encoding="utf-8")
    (task_dir / "overview_6panel.mp4").write_bytes(b"stale-video")
    batch.atomic_json(
        task_dir / "batch_task_summary.json",
        {
            **source_paths,
            "result_complete": True,
            "recording_enabled": True,
        },
    )
    args = SimpleNamespace(resume=True, recording=True, render_topdown=True)

    row = batch.run_task(task, args)

    assert row["resumed"] is True
    _assert_published(task_dir, source_paths, row)
    assert not (task_dir / "attempt_002").exists()


def test_reconcile_rows_publishes_recovered_attempt(tmp_path, monkeypatch):
    task = _task(tmp_path)
    task_dir = Path(task.task_dir)
    attempt_dir = task_dir / "attempt_001"
    source_paths = _write_attempt_artifacts(attempt_dir)

    def fake_parse_episode_result(attempt_task, _args):
        assert Path(attempt_task.task_dir) == attempt_dir
        return {
            **source_paths,
            "result_complete": True,
            "success": False,
            "task_success": False,
            "nav_success": False,
        }

    monkeypatch.setattr(batch, "parse_episode_result", fake_parse_episode_result)
    args = SimpleNamespace(recording=True, render_topdown=True)

    rows = batch.reconcile_rows_from_disk([task], [], args)

    assert len(rows) == 1
    _assert_published(task_dir, source_paths, rows[0])
    persisted = batch.read_json(task_dir / "batch_task_summary.json")
    _assert_published(task_dir, source_paths, persisted)
