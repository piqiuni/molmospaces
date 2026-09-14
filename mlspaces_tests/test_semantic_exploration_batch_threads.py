import sys
from types import SimpleNamespace

import pytest

from scripts.InteractiveNav import run_semantic_interaction_exploration_batch as batch


def arguments(monkeypatch, tmp_path, *extra):
    for name in batch.NATIVE_THREAD_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["batch", "--output-dir", str(tmp_path),
                                    "--house-inds", "1", "--dry-run", *extra])
    return batch.parse_args()


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_worker_native_thread_default_is_bounded(monkeypatch, tmp_path, threads):
    args = arguments(monkeypatch, tmp_path, "--native-threads-per-worker", str(threads))
    result = batch.run_scene(0, 1, args)
    assert result["native_thread_environment"] == dict.fromkeys(batch.NATIVE_THREAD_VARIABLES, str(threads))


def test_default_is_one_and_preserves_explicit_environment(monkeypatch, tmp_path):
    args = arguments(monkeypatch, tmp_path)
    assert args.native_threads_per_worker == 1
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    result = batch.run_scene(0, 1, args)
    assert result["native_thread_environment"] == {
        "OPENBLAS_NUM_THREADS": "3", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def test_zero_preserves_library_defaults(monkeypatch, tmp_path):
    args = arguments(monkeypatch, tmp_path, "--native-threads-per-worker", "0")
    result = batch.run_scene(0, 1, args)
    assert result["native_thread_environment"] == dict.fromkeys(batch.NATIVE_THREAD_VARIABLES)


def test_negative_thread_count_is_rejected(monkeypatch, tmp_path):
    arguments(monkeypatch, tmp_path, "--native-threads-per-worker", "-1")
    with pytest.raises(ValueError, match="native-threads-per-worker"):
        batch.main()


def test_thread_settings_reach_child_process(monkeypatch, tmp_path):
    args = arguments(monkeypatch, tmp_path)
    args.dry_run = False
    captured = {}

    def launch(command, **kwargs):
        captured.update(kwargs["env"])
        return SimpleNamespace(pid=1, wait=lambda **_kwargs: 0)

    monkeypatch.setattr(batch.subprocess, "Popen", launch)
    monkeypatch.setattr(batch, "monitor_scene_memory", lambda *_args: None)
    result = batch.run_scene(0, 1, args)
    for name in batch.NATIVE_THREAD_VARIABLES:
        assert captured[name] == result["native_thread_environment"][name] == "1"
