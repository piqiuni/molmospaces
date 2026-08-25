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


def _recovery_args(tmp_path):
    output_dir = tmp_path / "round"
    output_dir.mkdir(parents=True)
    benchmark = tmp_path / "mixed.json"
    benchmark.write_text("[]\n", encoding="utf-8")
    return SimpleNamespace(
        output_dir=output_dir,
        benchmark=benchmark,
        runner=SCRIPT_PATH,
        runner_shell="bash",
        workers=1,
        base_master_port=12600,
        max_steps=2000,
        step_budget_mode="fixed",
        semantic_attribute_request_timeout_s=20.0,
        ros_command_starvation_timeout_s=60.0,
        ros_observation_turn_multiplier=1.5,
        scene_timeout_s=7200.0,
        fast_eval=True,
        model_endpoints=None,
        mujoco_egl_devices=None,
        semantic_model_env_file=None,
        resource_telemetry=False,
        resource_sample_interval_s=2.0,
        resume=True,
    )


def _recovery_plan(args, episode_index=0):
    return batch.EpisodePlan(
        ordinal=episode_index,
        worker_id=0,
        episode_index=episode_index,
        master_port=12600 + episode_index,
        task_dir=args.output_dir / f"episode_{episode_index:04d}",
    )


def _write_recovery_attempt(
    plan,
    args,
    *,
    attempt_name="attempt_001",
    result_index=None,
    result_signature="run-signature",
    manifest_signature="run-signature",
    manifest_indices=None,
):
    result_index = plan.episode_index if result_index is None else result_index
    manifest_indices = [plan.episode_index] if manifest_indices is None else manifest_indices
    attempt_dir = plan.task_dir / attempt_name
    result_path = (
        attempt_dir
        / "eval"
        / "episodes"
        / f"{result_index:04d}_case"
        / "episode_result.json"
    )
    result_path.parent.mkdir(parents=True)
    planned_record = batch.planned_invocation_record(plan, args)
    assert planned_record is not None
    (attempt_dir / batch.PLANNED_INVOCATION_FILENAME).write_text(
        json.dumps(planned_record),
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "run_signature": result_signature,
                "result": {
                    "status": "complete",
                    "episode_index": result_index,
                    "success": False,
                    "task_success": False,
                    "nav_success": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (attempt_dir / "eval" / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_signature": manifest_signature,
                "benchmark_sha256": batch.sha256_file(args.benchmark),
                "episode_indices": manifest_indices,
                "evaluation_config": {
                    "episode_indices": manifest_indices,
                    "max_steps": args.max_steps,
                    "step_budget_mode": args.step_budget_mode,
                    "ros_command_starvation_timeout_s": args.ros_command_starvation_timeout_s,
                    "ros_observation_turn_multiplier": args.ros_observation_turn_multiplier,
                },
            }
        ),
        encoding="utf-8",
    )
    plan.task_dir.mkdir(parents=True, exist_ok=True)
    (plan.task_dir / "batch_task.log").write_text(
        f"[start] attempt={attempt_name} start command=bash {SCRIPT_PATH} {attempt_dir} {plan.episode_index}\n",
        encoding="utf-8",
    )
    return attempt_dir, result_path


def test_fast_eval_resume_requires_matching_planned_invocation_without_an_mp4(tmp_path):
    args = _recovery_args(tmp_path)
    plan = _recovery_plan(args)
    task_dir = plan.task_dir
    attempt_dir = task_dir / "attempt_001"
    result_path = attempt_dir / "eval" / "episodes" / "0000_case" / "episode_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps({"status": "complete", "result": {"status": "complete"}}),
        encoding="utf-8",
    )
    planned_record = batch.planned_invocation_record(plan, args)
    assert planned_record is not None
    (task_dir / "batch_task_summary.json").write_text(
        json.dumps(
            {
                "completed": True,
                "fast_eval": True,
                "episode_result_path": str(result_path),
                # Deliberately no six_panel_video: this is the fast contract.
                **planned_record,
            }
        ),
        encoding="utf-8",
    )

    assert batch.existing_completed_summary(plan, args)
    changed = SimpleNamespace(**vars(args))
    changed.fast_eval = False
    assert batch.existing_completed_summary(plan, changed) is None


def test_resume_refuses_completed_summary_when_planned_contract_changes(tmp_path, monkeypatch):
    args = _recovery_args(tmp_path)
    plan = _recovery_plan(args)
    task_dir = plan.task_dir
    attempt_dir = task_dir / "attempt_001"
    result_path = attempt_dir / "eval" / "episodes" / "0000_case" / "episode_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps({"status": "complete", "result": {"status": "complete"}}),
        encoding="utf-8",
    )
    planned_record = batch.planned_invocation_record(plan, args)
    assert planned_record is not None
    (task_dir / "batch_task_summary.json").write_text(
        json.dumps(
            {
                "completed": True,
                "fast_eval": True,
                "episode_result_path": str(result_path),
                **planned_record,
            }
        ),
        encoding="utf-8",
    )
    assert batch.existing_completed_summary(plan, args)

    changed_steps = SimpleNamespace(**vars(args))
    changed_steps.max_steps = 1500
    assert batch.existing_completed_summary(plan, changed_steps) is None

    changed_timeout = SimpleNamespace(**vars(args))
    changed_timeout.semantic_attribute_request_timeout_s = 30.0
    assert batch.existing_completed_summary(plan, changed_timeout) is None

    args.benchmark.write_text("[\"different-frozen-benchmark\"]\n", encoding="utf-8")
    assert batch.existing_completed_summary(plan, args) is None
    args.benchmark.write_text("[]\n", encoding="utf-8")
    assert batch.existing_completed_summary(plan, args)

    monkeypatch.setenv("POLICY", "different_policy")
    assert batch.existing_completed_summary(plan, args) is None


def test_resume_refuses_completed_summary_when_ros_runtime_contract_changes(tmp_path, monkeypatch):
    args = _recovery_args(tmp_path)
    plan = _recovery_plan(args)
    runtime_file = tmp_path / "runtime.launch"
    runtime_file.write_text("<launch/>\n", encoding="utf-8")
    monkeypatch.setattr(batch, "_ROS_RUNTIME_PROTOCOL_FILES", (runtime_file,))
    task_dir = plan.task_dir
    result_path = task_dir / "attempt_001" / "eval" / "episodes" / "0000_case" / "episode_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps({"status": "complete", "result": {"status": "complete"}}),
        encoding="utf-8",
    )
    planned_record = batch.planned_invocation_record(plan, args)
    assert planned_record is not None
    (task_dir / "batch_task_summary.json").write_text(
        json.dumps({"completed": True, "fast_eval": True, "episode_result_path": str(result_path), **planned_record}),
        encoding="utf-8",
    )
    assert batch.existing_completed_summary(plan, args)
    runtime_file.write_text("<launch><arg name='changed'/></launch>\n", encoding="utf-8")
    assert batch.existing_completed_summary(plan, args) is None


def test_planned_invocation_tracks_effective_inherited_runtime(tmp_path, monkeypatch):
    args = _recovery_args(tmp_path)
    plan = _recovery_plan(args)
    conda_a = tmp_path / "conda_a"
    conda_b = tmp_path / "conda_b"
    for conda_dir in (conda_a, conda_b):
        python = conda_dir / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(batch, "infer_conda_prefix", lambda: None)
    monkeypatch.delenv("PYTHON_BIN", raising=False)
    monkeypatch.setenv("CONDA_ENV", str(conda_a))

    first = batch.planned_invocation_record(plan, args)
    assert first is not None
    assert first["planned_invocation"]["runtime_conda"] == str(conda_a.resolve())
    assert first["planned_invocation"]["runtime_python"] == str(
        (conda_a / "bin" / "python").resolve()
    )

    monkeypatch.setenv("CONDA_ENV", str(conda_b))
    second = batch.planned_invocation_record(plan, args)
    assert second is not None
    assert second["planned_invocation_signature"] != first["planned_invocation_signature"]


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


def test_worker_environment_keeps_ros_and_all_caches_under_output_dir(tmp_path, monkeypatch):
    output_dir = tmp_path / "round"
    attempt_dir = output_dir / "episode_0000" / "attempt_001"
    args = SimpleNamespace(
        output_dir=output_dir,
        benchmark=tmp_path / "mixed.json",
        max_steps=2000,
        step_budget_mode="fixed",
        semantic_attribute_request_timeout_s=20.0,
        ros_command_starvation_timeout_s=60.0,
        ros_observation_turn_multiplier=1.5,
        fast_eval=True,
        conda_env=None,
        python_bin=None,
        model_endpoints=None,
        semantic_model_env_file=None,
        mujoco_egl_devices=None,
    )
    plan = batch.EpisodePlan(
        ordinal=0,
        worker_id=0,
        episode_index=0,
        master_port=12600,
        task_dir=output_dir / "episode_0000",
    )
    monkeypatch.setattr(batch, "infer_conda_prefix", lambda: None)

    environment = batch.make_environment(args, plan, attempt_dir)

    assert environment["TMPDIR"] == str(attempt_dir / "tmp")
    assert environment["STEP_BUDGET_MODE"] == "fixed"
    assert environment["SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S"] == "20.0"
    assert environment["ROS_COMMAND_STARVATION_TIMEOUT_S"] == "60.0"
    assert environment["ROS_OBSERVATION_TURN_MULTIPLIER"] == "1.5"
    assert environment["ROS_HOME"] == str(attempt_dir / "ros_home")
    assert environment["ROS_LOG_DIR"] == str(attempt_dir / "ros_home" / "log")
    assert environment["INTERACTIVE_NAV_SCENE_MIRROR"] == str(attempt_dir / "scene_mirror")
    assert environment["HF_HOME"] == str(output_dir / "_runtime_cache" / "hf")
    assert environment["TORCH_HOME"] == str(output_dir / "_runtime_cache" / "torch")
    assert environment["CUDA_CACHE_PATH"] == str(output_dir / "_runtime_cache" / "cuda")
    assert all(str(output_dir) in environment[key] for key in (
        "TMPDIR", "ROS_HOME", "ROS_LOG_DIR", "INTERACTIVE_NAV_SCENE_MIRROR",
        "HF_HOME", "TORCH_HOME", "CUDA_CACHE_PATH"
    ))


def test_summary_marks_unreported_plans_as_incomplete_and_not_successful(tmp_path):
    output_dir = tmp_path / "round"
    output_dir.mkdir()
    args = SimpleNamespace(
        output_dir=output_dir,
        benchmark=tmp_path / "mixed.json",
        runner=SCRIPT_PATH,
        runner_shell="bash",
        workers=2,
        base_master_port=12600,
        max_steps=2000,
        step_budget_mode="fixed",
        semantic_attribute_request_timeout_s=20.0,
        ros_command_starvation_timeout_s=60.0,
        ros_observation_turn_multiplier=1.5,
        scene_timeout_s=7200.0,
        fast_eval=True,
        model_endpoints=None,
        mujoco_egl_devices=None,
        semantic_model_env_file=None,
        resource_telemetry=True,
        resource_sample_interval_s=2.0,
        resume=False,
    )
    plans = [
        batch.EpisodePlan(0, 0, 0, 12600, output_dir / "episode_0000"),
        batch.EpisodePlan(1, 1, 1, 12601, output_dir / "episode_0001"),
    ]
    reported_result = {
        "episode_index": 0,
        "completed": True,
        "success": True,
        "task_success": True,
        "nav_success": True,
    }

    aggregate = batch.write_summary(args, plans, [reported_result])
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert aggregate["planned_episode_count"] == 2
    assert aggregate["reported_episode_count"] == 1
    assert aggregate["missing_episode_count"] == 1
    assert aggregate["planned_minus_reported_episode_count"] == 1
    assert aggregate["failed_or_incomplete_episode_count"] == 1
    assert summary["aggregate"] == aggregate
    assert [row["episode_index"] for row in summary["episodes"]] == [0, 1]
    missing = summary["missing_episode_reports"]
    assert len(missing) == 1
    assert missing[0]["episode_index"] == 1
    assert missing[0]["report_status"] == "missing"
    assert missing[0]["completed"] is False
    assert missing[0]["success"] is False
    assert missing[0]["scoring_eligible"] is False


def test_summary_recovers_one_valid_current_attempt_and_counts_it_formally(tmp_path):
    args = _recovery_args(tmp_path)
    plan = _recovery_plan(args)
    attempt_dir, result_path = _write_recovery_attempt(plan, args)

    results = []
    aggregate = batch.write_summary(
        args,
        [plan],
        results,
        recover_missing_task_summaries=True,
    )

    assert len(results) == 1
    recovered = results[0]
    assert recovered["completed"] is True
    assert recovered["recovered_batch_task_summary"] is True
    assert recovered["run_signature"] == "run-signature"
    assert recovered["episode_result_path"] == str(result_path)
    assert recovered["attempt_dir"] == str(attempt_dir)
    assert aggregate["completed_episode_count"] == 1
    assert aggregate["failed_or_incomplete_episode_count"] == 0
    assert aggregate["missing_episode_count"] == 0
    assert (plan.task_dir / "batch_task_summary.json").is_file()


def test_recovery_rejects_mismatched_signature_or_index(tmp_path):
    args = _recovery_args(tmp_path)
    plan = _recovery_plan(args)
    _write_recovery_attempt(
        plan,
        args,
        result_signature="result-signature",
        manifest_signature="different-signature",
    )

    assert batch.recover_missing_task_summary(plan, args) is None
    assert not (plan.task_dir / "batch_task_summary.json").exists()

    args = _recovery_args(tmp_path / "index_mismatch")
    plan = _recovery_plan(args)
    _write_recovery_attempt(plan, args, result_index=1)
    assert batch.recover_missing_task_summary(plan, args) is None


def test_recovery_rejects_attempt_from_different_planned_contract(tmp_path):
    args = _recovery_args(tmp_path)
    plan = _recovery_plan(args)
    _write_recovery_attempt(plan, args)

    changed = SimpleNamespace(**vars(args))
    changed.scene_timeout_s = 3600.0
    assert batch.recover_missing_task_summary(plan, changed) is None
    assert not (plan.task_dir / "batch_task_summary.json").exists()


def test_recovery_rejects_ambiguous_current_attempt_and_stale_fallback(tmp_path):
    args = _recovery_args(tmp_path)
    plan = _recovery_plan(args)
    _write_recovery_attempt(plan, args)
    second = plan.task_dir / "attempt_001" / "eval" / "episodes" / "0000_other" / "episode_result.json"
    second.parent.mkdir(parents=True)
    second.write_text(
        json.dumps({"status": "complete", "run_signature": "run-signature", "result": {"status": "complete", "episode_index": 0}}),
        encoding="utf-8",
    )
    assert batch.recover_missing_task_summary(plan, args) is None

    args = _recovery_args(tmp_path / "new_attempt")
    plan = _recovery_plan(args)
    _write_recovery_attempt(plan, args, attempt_name="attempt_001")
    (plan.task_dir / "attempt_002").mkdir()
    assert batch.recover_missing_task_summary(plan, args) is None
