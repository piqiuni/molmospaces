import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).with_name("run_interactive_nav_v3_ros_eval_batch.py")
SPEC = importlib.util.spec_from_file_location("run_interactive_nav_v3_ros_eval_batch", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
batch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batch
SPEC.loader.exec_module(batch)


def test_fast_eval_resume_accepts_complete_result_without_an_mp4(tmp_path):
    task_dir = tmp_path / "episode_0000"
    attempt_dir = task_dir / "attempt_001"
    result_path = attempt_dir / "eval" / "episodes" / "0000_case" / "episode_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps({"status": "complete", "result": {"status": "complete"}}),
        encoding="utf-8",
    )
    (task_dir / "batch_task_summary.json").write_text(
        json.dumps(
            {
                "completed": True,
                "fast_eval": True,
                "episode_result_path": str(result_path),
                # Deliberately no six_panel_video: this is the fast contract.
            }
        ),
        encoding="utf-8",
    )

    assert batch.existing_completed_summary(task_dir, SimpleNamespace(fast_eval=True))
    assert batch.existing_completed_summary(task_dir, SimpleNamespace(fast_eval=False)) is None


def test_derived_endpoint_dotenv_preserves_base_settings_and_overrides_endpoint(tmp_path):
    source = tmp_path / "base.env"
    source.write_text(
        "SEMANTIC_MODEL_NAME=qwen-local\nSEMANTIC_MODEL_TIMEOUT_S=30\n"
        "SEMANTIC_MODEL_ENDPOINT=http://old.invalid/v1\n",
        encoding="utf-8",
    )
    attempt = tmp_path / "attempt_001"
    attempt.mkdir()

    derived = batch.derive_episode_model_env_file(attempt, source, "http://127.0.0.1:8001/v1")

    contents = derived.read_text(encoding="utf-8")
    assert "SEMANTIC_MODEL_NAME=qwen-local" in contents
    assert "SEMANTIC_MODEL_TIMEOUT_S=30" in contents
    assert contents.rstrip().endswith("SEMANTIC_MODEL_ENDPOINT=http://127.0.0.1:8001/v1")


def test_worker_endpoint_and_egl_assignment_are_round_robin():
    args = SimpleNamespace(
        model_endpoints=["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
        mujoco_egl_devices=["0", "1"],
    )

    assert [batch.assigned_model_endpoint(worker, args) for worker in range(4)] == [
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1:8001/v1",
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1:8001/v1",
    ]
    assert [batch.assigned_mujoco_egl_device(worker, args) for worker in range(4)] == [
        "0",
        "1",
        "0",
        "1",
    ]
