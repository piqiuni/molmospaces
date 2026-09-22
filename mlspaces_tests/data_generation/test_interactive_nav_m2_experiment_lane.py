from collections import Counter
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts.InteractiveNav import run_m2_experiment_lane as lane


REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "scripts/InteractiveNav/configs/evaluation/m2_online_mixed0_29_8arm.json"


def _manifest(tmp_path=None):
    result = json.loads(TEMPLATE.read_text())
    if tmp_path is not None:
        for entry in result["lanes"]:
            entry["output_dir"] = str(tmp_path / "lanes" / str(entry["id"]))
    return result


def test_fixed_plan_is_240_unique_jobs_balanced_and_scene_paired():
    result = lane.normalize_manifest(_manifest())
    jobs = result["planned_jobs"]
    assert len(jobs) == len({job["job_id"] for job in jobs}) == 240
    assert Counter(job["lane_id"] for job in jobs) == {0: 80, 1: 80, 2: 80}
    assert Counter(job["arm"] for job in jobs) == {f"G{i}": 30 for i in range(8)}
    for index in range(30):
        block = [job for job in jobs if job["mixed_index"] == index]
        assert {job["lane_id"] for job in block} == {index % 3}
        assert {job["episode_index"] for job in block} == {2000 + index}
    assert lane.normalize_manifest(result) == result
    assert lane.normalize_manifest(_manifest())["planned_jobs"] == jobs


@pytest.mark.parametrize("fault", ["wrong_budget", "wrong_profile", "overlap_ports", "same_gpu", "root_cache", "unknown_env", "remote_endpoint", "wrong_job"])
def test_rejects_unsafe_or_unpaired_manifest(fault):
    result = _manifest()
    if fault == "wrong_budget": result["worker_count"] = 30
    if fault == "wrong_profile": result["arms"][0]["environment"]["SEMANTIC_M2_CONTEXT_PROFILE"] = "public_facts_v2"
    if fault == "overlap_ports": result["lanes"][1]["base_master_port"] = result["lanes"][0]["base_master_port"] + 1
    if fault == "same_gpu": result["lanes"][1]["egl_device_id"] = "0"
    if fault == "root_cache": result["common_environment"]["MLSPACES_CACHE_DIR"] = "/tmp/cache"
    if fault == "unknown_env": result["common_environment"]["HOME"] = "/home/ldl"
    if fault == "remote_endpoint": result["lanes"][0]["model_endpoint"] = "http://example.invalid/v1"
    if fault == "wrong_job": result["planned_jobs"] = []
    with pytest.raises(ValueError): lane.normalize_manifest(result)


def test_model_overrides_force_qwen_and_do_not_remap_egl():
    manifest = lane.normalize_manifest(_manifest())
    current = manifest["lanes"][1]
    current["environment"]["SEMANTIC_M2_ENDPOINT"] = "http://wrong.invalid/v1"
    overrides = lane.arm_environment(manifest, current, manifest["arms"][0])
    assert overrides["SEMANTIC_MODEL_ENDPOINT"] == overrides["SEMANTIC_M2_ENDPOINT"] == "http://127.0.0.1:8000/v1"
    assert overrides["SEMANTIC_M2_MODE"] == "http"
    assert overrides["SEMANTIC_M2_PROTOCOL"] == "openai_chat"
    assert overrides["SEMANTIC_M2_TIMEOUT_S"] == "120"
    assert overrides["SEMANTIC_M2_MODEL_NAME"] == lane.MODEL_NAME
    assert current["egl_device_id"] == "1"
    assert all(overrides[key] == "1" for key in lane.THREAD_KEYS)


def test_duplicate_dotenv_keys_removed_and_wrapper_overrides_inherited_default(tmp_path):
    manifest = lane.normalize_manifest(_manifest())
    overrides = lane.arm_environment(manifest, manifest["lanes"][0], manifest["arms"][3])
    source = "SEMANTIC_M2_TIMEOUT_S=30\nexport SEMANTIC_M2_TIMEOUT_S=55\nSEMANTIC_M2_ENDPOINT=http://old.invalid/v1\nUNCHANGED=value\n"
    derived = lane.derived_env_text(source, overrides)
    assert derived.count("SEMANTIC_M2_TIMEOUT_S=") == 1
    assert "http://old.invalid" not in derived
    env_path = tmp_path / "derived.env"
    env_path.write_text(derived)
    snapshot = tmp_path / "probe.sh"
    snapshot.write_text(f'exec {sys.executable} -c \'import os,json;print(json.dumps({{k:os.environ[k] for k in ["SEMANTIC_M2_TIMEOUT_S","SEMANTIC_M2_ENDPOINT","SEMANTIC_M2_PROMPT_FILE"]}}))\'\n')
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(lane.wrapper_text(snapshot, overrides))
    environment = {**os.environ, "SEMANTIC_M2_TIMEOUT_S": "30", "SEMANTIC_MODEL_ENV_FILE": str(env_path)}
    actual = json.loads(subprocess.check_output(["bash", str(wrapper)], env=environment, text=True))
    assert actual["SEMANTIC_M2_TIMEOUT_S"] == "120"
    assert actual["SEMANTIC_M2_ENDPOINT"] == "http://127.0.0.1:8000/v1"
    assert actual["SEMANTIC_M2_PROMPT_FILE"] == overrides["SEMANTIC_M2_PROMPT_FILE"]


def test_real_dotenv_loader_and_model_merge_use_expected_profile(tmp_path, monkeypatch):
    path = REPO / "Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts/semantic_decision_py_pkg/env_config.py"
    spec = importlib.util.spec_from_file_location("lane_env_config_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = lane.normalize_manifest(_manifest())
    overrides = lane.arm_environment(manifest, manifest["lanes"][0], manifest["arms"][0])
    env_file = tmp_path / "arm.env"
    env_file.write_text(lane.derived_env_text("SEMANTIC_M2_TIMEOUT_S=30\nSEMANTIC_M2_CONTEXT_PROFILE=public_facts_v2\n", overrides))
    for key in overrides: monkeypatch.setenv(key, overrides[key])
    module.load_env_file(env_file, override=False)
    result = module.apply_model_env_overrides({"timeout_s": 30})
    assert result["timeout_s"] == 120
    assert result["context_profile"] == "historical_compat_v1"
    assert result["candidate_pool_mode"] == "legacy"
    assert result["recent_decision_limit"] == 8


def test_algorithm_failure_never_enters_failure_queue():
    assert lane.classify_incomplete({"completed": True, "success": False}) is None
    unknown = lane.classify_incomplete({"completed": False, "runner_exit_code": 1})
    assert unknown["classification"] == "incomplete_unclassified"
    assert unknown["retry_authorized"] is False
    clear = lane.classify_incomplete({"completed": False, "error": "ModuleNotFoundError: test"})
    assert clear["classification"] == "confirmed_infrastructure"


def test_status_keeps_all_planned_jobs_and_per_arm_counts():
    manifest = lane.normalize_manifest(_manifest())
    selected = manifest["lanes"][0]
    jobs = [job for job in manifest["planned_jobs"] if job["lane_id"] == 0]
    results = {jobs[0]["job_id"]: {"completed": True, "success": False},
               jobs[1]["job_id"]: {"completed": False, "runner_exit_code": 1}}
    active = {0: jobs[2]}
    status = lane.lane_status(manifest, selected, jobs, results, active, "running", time.monotonic())
    assert status["planned"] == 80 and status["reported"] == 2
    assert status["queued"] == 77 and status["completed"] == status["failed"] == 1
    assert len(status["failure_queue"]) == 1
    assert all(stat["planned"] == 10 for stat in status["per_arm"].values())


def test_prepare_snapshots_without_reading_source_dotenv_and_hash_guard(tmp_path, monkeypatch):
    result = _manifest(tmp_path / "experiment")
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text("[]")
    result["benchmark"] = str(benchmark)
    result["benchmark_sha256"] = lane._sha(benchmark)
    result["launcher_config"]["semantic_model_env_file"] = "/not/a/real/dotenv"
    template = tmp_path / "template.json"
    template.write_text(json.dumps(result))
    dummy_source = tmp_path / "source.py"
    dummy_source.write_text("# frozen\n")
    monkeypatch.setattr(lane, "_source_files", lambda: [dummy_source])
    root = tmp_path / "experiment"
    frozen = lane.prepare_manifest(template, root)
    assert len(frozen["planned_jobs"]) == 240
    assert (root / "manifest.json").exists()
    assert len(list((root / "prompts").iterdir())) == 8
    lane.verify_frozen_manifest(frozen)
    dummy_source.write_text("# changed\n")
    with pytest.raises(ValueError, match="source hash mismatch"): lane.verify_frozen_manifest(frozen)
    with pytest.raises(FileExistsError): lane.prepare_manifest(template, root)


def test_model_identity_and_context_capacity_validation():
    result = lane.validate_model_inventory({"data": [{"id": lane.MODEL_NAME, "max_model_len": 16384}]})
    assert result["healthy"] is True
    with pytest.raises(RuntimeError): lane.validate_model_inventory({"data": [{"id": "wrong"}]})
    with pytest.raises(RuntimeError): lane.validate_model_inventory({"data": [{"id": lane.MODEL_NAME, "max_model_len": 10240}]})


def test_official_dry_run_does_not_create_output_or_start_services(tmp_path):
    manifest = _manifest(tmp_path / "experiment")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    completed = subprocess.run([sys.executable, str(REPO / "scripts/InteractiveNav/run_benchmark_eval.py"),
                                "--experiment-manifest", str(path), "--experiment-lane", "1", "--dry-run"],
                               capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)
    assert len(result["jobs"]) == 80
    assert result["config"]["mujoco_egl_devices"] == ["1"]
    assert result["config"]["model_endpoints"] == ["http://127.0.0.1:8000/v1"]
    assert result["start_qwen"] is False
    assert not (tmp_path / "experiment").exists()
