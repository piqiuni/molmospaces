"""Fast tests for the native nav-to-object lease scheduler.

These tests deliberately avoid importing MuJoCo or starting ROS.  They exercise
only benchmark manifest parsing, SQLite coordination, and result discovery.
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.InteractiveNav import nav_to_obj_batch_manager as manager  # noqa: E402


def _write_benchmark(path: Path, count: int = 4) -> Path:
    benchmark_dir = path / "benchmark"
    benchmark_dir.mkdir()
    payload = [
        {
            "house_index": 10 + index,
            "source": {"traj_key": "repeated_traj_key"},
            "task": {"pickup_obj_name": f"{'laptop' if index % 2 else 'vase'}_asset_{index}"},
        }
        for index in range(count)
    ]
    (benchmark_dir / "benchmark.json").write_text(json.dumps(payload), encoding="utf-8")
    return benchmark_dir


def _initialize(
    tmp_path: Path,
    *,
    selected: list[str] | None = None,
    max_attempts: int = 2,
    command_template: str = "echo {episode_idx}",
    worker_env: list[str] | None = None,
) -> Path:
    benchmark_dir = _write_benchmark(tmp_path)
    run_root = tmp_path / "run"
    argv = [
        "init",
        "--benchmark-dir",
        str(benchmark_dir),
        "--run-root",
        str(run_root),
        "--command-template",
        command_template,
        "--seed",
        "17",
        "--max-attempts-per-episode",
        str(max_attempts),
        "--worker-env",
        "HOME=/remote/home",
    ]
    for entry in worker_env or []:
        argv.extend(["--worker-env", entry])
    if selected is not None:
        argv.extend(["--episode-indices", *selected])
    assert manager.main(argv) == 0
    return run_root


def test_init_uses_global_subset_and_records_episode_target_metadata(tmp_path: Path) -> None:
    run_root = _initialize(tmp_path, selected=["3,2"])

    config = json.loads((run_root / manager.RUN_CONFIG_FILENAME).read_text(encoding="utf-8"))
    manifest = json.loads((run_root / manager.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    _root, loaded_config, ledger = manager._load_run_config(run_root)

    assert config["benchmark_total_episode_count"] == 4
    assert config["selected_episode_indices"] == [3, 2]
    assert config["worker_environment"] == {"HOME": "/remote/home"}
    assert [row["global_episode_idx"] for row in manifest["episodes"]] == [3, 2]
    assert all(row["benchmark_sha256"] == config["benchmark_sha256"] for row in manifest["episodes"])
    assert {row["target_type"] for row in ledger.plan(count=10, max_attempts=2)} == {
        "laptop",
        "vase",
    }
    assert loaded_config["episode_count"] == 2


def test_init_scene_count_selects_distinct_houses_deterministically(tmp_path: Path) -> None:
    benchmark_dir = _write_benchmark(tmp_path, count=4)
    run_root = tmp_path / "scene_count_run"

    assert manager.main(
        [
            "init",
            "--benchmark-dir",
            str(benchmark_dir),
            "--run-root",
            str(run_root),
            "--command-template",
            "echo {episode_idx}",
            "--seed",
            "17",
            "--scene-count",
            "3",
        ]
    ) == 0

    manifest = json.loads((run_root / manager.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert len(manifest["episodes"]) == 3
    assert len({row["house_index"] for row in manifest["episodes"]}) == 3


def test_init_rejects_scene_count_with_explicit_episode_indices(tmp_path: Path) -> None:
    benchmark_dir = _write_benchmark(tmp_path)
    with pytest.raises(ValueError, match="cannot be used together"):
        manager.main(
            [
                "init",
                "--benchmark-dir",
                str(benchmark_dir),
                "--run-root",
                str(tmp_path / "invalid"),
                "--command-template",
                "echo {episode_idx}",
                "--scene-count",
                "2",
                "--episode-indices",
                "0",
            ]
        )


def test_concurrent_claims_are_unique_and_terminal_results_are_deduplicated(tmp_path: Path) -> None:
    run_root = _initialize(tmp_path, selected=["0", "1"])
    _root, _config, ledger = manager._load_run_config(run_root)

    def claim(worker_number: int) -> manager.Claim | None:
        return manager.EpisodeLedger(ledger.database_path).claim_next(
            worker_id=f"worker-{worker_number}",
            lease_seconds=60.0,
            max_attempts=2,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        claims = [claim for claim in executor.map(claim, range(4)) if claim is not None]

    assert len(claims) == 2
    assert len({claim.episode_idx for claim in claims}) == 2
    assert len({claim.claim_token for claim in claims}) == 2
    assert all(claim.benchmark_sha256 for claim in claims)
    for claim in claims:
        assert ledger.finish(
            claim,
            status="completed",
            return_code=0,
            official_success=claim.target_type == "laptop",
            result_summary_path=None,
            error_message=None,
        )

    status = ledger.status()
    assert status["selected_episode_count"] == 2
    assert status["terminal_episode_count"] == 2
    assert status["leased_episode_count"] == 0
    assert status["pending_episode_count"] == 0
    assert status["official_successes"] == 1


def test_expired_lease_is_reclaimed_once_with_a_new_attempt(tmp_path: Path) -> None:
    run_root = _initialize(tmp_path, selected=["2"], max_attempts=2)
    _root, _config, ledger = manager._load_run_config(run_root)

    first = ledger.claim_next(worker_id="lost-worker", lease_seconds=1.0, max_attempts=2, now=100.0)
    assert first is not None
    assert ledger.reclaim_expired(now=102.0, max_attempts=2) == [2]
    second = ledger.claim_next(worker_id="replacement", lease_seconds=60.0, max_attempts=2, now=103.0)

    assert second is not None
    assert second.episode_idx == first.episode_idx
    assert second.attempt == 2
    assert second.claim_token != first.claim_token


def test_template_and_native_summary_use_per_episode_target_context(tmp_path: Path) -> None:
    claim = manager.Claim(
        benchmark_sha256="a" * 64,
        episode_idx=7,
        house_index=31,
        source_traj_key="traj_7",
        pickup_obj_name="vase_asset_7",
        target_type="vase",
        attempt=1,
        worker_id="worker-0",
        claim_token="token1234567890",
        claimed_at=1.0,
        lease_expires_at=2.0,
    )
    run_config = {
        "launcher": "/tmp/launcher.sh",
        "repo_root": "/tmp/repo",
        "benchmark_dir": "/tmp/benchmark",
        "command_template": "{launcher} {episode_idx} {house_index} {target_type} {output_dir}",
    }
    command = manager._format_command(
        run_config,
        claim=claim,
        output_dir=tmp_path / "attempt" / "output",
        ros_master_uri="http://127.0.0.1:11601",
    )
    assert command[:4] == ["/tmp/launcher.sh", "7", "31", "vase"]

    official_dir = tmp_path / "attempt" / "output" / "NativeNavToObjEvalConfig" / "stamp"
    official_dir.mkdir(parents=True)
    summary_path = official_dir / "native_eval_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "official_eval_output_dir": str(official_dir),
                "success_count": 1,
                "total_count": 1,
            }
        ),
        encoding="utf-8",
    )
    found_path, summary, error = manager._find_native_summary(tmp_path / "attempt")
    assert found_path == summary_path
    assert summary is not None and summary["success_count"] == 1
    assert error is None


def test_worker_reserves_each_ros_master_uri_for_one_live_session(tmp_path: Path) -> None:
    run_root = _initialize(tmp_path, selected=["0"])
    _root, _config, ledger = manager._load_run_config(run_root)
    uri = "http://127.0.0.1:11605"

    ledger.reserve_worker_slot(
        worker_id="first-worker",
        session_token="first-session",
        ros_master_uri=uri,
        lease_seconds=60.0,
    )
    with pytest.raises(RuntimeError, match="already reserved"):
        ledger.reserve_worker_slot(
            worker_id="same-visible-worker-id-is-not-enough",
            session_token="second-session",
            ros_master_uri=uri,
            lease_seconds=60.0,
        )
    assert ledger.heartbeat_worker_slot(
        session_token="first-session",
        ros_master_uri=uri,
        lease_seconds=60.0,
    )
    assert ledger.status()["active_worker_slot_count"] == 1
    ledger.release_worker_slot(session_token="first-session", ros_master_uri=uri)
    ledger.reserve_worker_slot(
        worker_id="replacement-worker",
        session_token="replacement-session",
        ros_master_uri=uri,
        lease_seconds=60.0,
    )
    ledger.release_worker_slot(session_token="replacement-session", ros_master_uri=uri)


def test_worker_refuses_benchmark_drift_before_claiming_an_episode(tmp_path: Path) -> None:
    run_root = _initialize(tmp_path, selected=["0"])
    benchmark_path = tmp_path / "benchmark" / "benchmark.json"
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    payload[0]["house_index"] = 999
    benchmark_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Benchmark content changed after init"):
        manager.main(
            [
                "worker",
                "--run-root",
                str(run_root),
                "--worker-id",
                "drift-checker",
                "--max-episodes",
                "1",
            ]
        )
    _root, _config, ledger = manager._load_run_config(run_root)
    assert ledger.status()["pending_episode_count"] == 1


def test_launcher_start_error_is_saved_as_a_failed_attempt_without_heartbeat_crash(
    tmp_path: Path,
) -> None:
    run_root = _initialize(
        tmp_path,
        selected=["0"],
        command_template="/definitely/missing/native-nav-launcher {output_dir}",
    )

    assert manager.main(
        [
            "worker",
            "--run-root",
            str(run_root),
            "--worker-id",
            "missing-launcher-worker",
            "--max-episodes",
            "1",
        ]
    ) == 0

    result_path = next((run_root / "episodes").rglob("batch_result.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["failure_kind"] == "launcher_exception"
    assert "FileNotFoundError" in result["error_message"]
    _root, _config, ledger = manager._load_run_config(run_root)
    assert ledger.status()["counts"]["failed"] == 1


def test_worker_passes_house_and_target_context_to_isolated_native_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROS_TASK_HORIZON", "777")
    monkeypatch.setenv("FILTER_MISSING_SCENE_OBJECTS", "true")
    monkeypatch.setenv("SEMANTIC_DECISION_ENV_FILE", "/stale/secret.env")
    monkeypatch.setenv("DEBUG_DIR", "/shared/debug")
    monkeypatch.setenv("TMPDIR", "/shared/tmp")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-gpu")
    fake_launcher = tmp_path / "fake_native_launcher.sh"
    fake_launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "output_dir=$1\n"
        "official_dir=${output_dir}/NativeNavToObjEvalConfig/fake\n"
        "mkdir -p ${official_dir}\n"
        "printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\\n' \"${ROS_HOUSE_IND}\" "
        "\"${ROS_TARGET_TYPES}\" \"${EPISODE_IDX}\" \"${ROS_MASTER_URI}\" "
        "\"${ROS_TASK_HORIZON-unset}\" \"${FILTER_MISSING_SCENE_OBJECTS}\" "
        "\"${DEBUG_DIR}\" \"${TMPDIR}\" \"${SEMANTIC_DECISION_ENV_FILE-unset}\" "
        "\"${CUDA_VISIBLE_DEVICES-unset}\"\n"
        "printf '{\"official_eval_output_dir\":\"%s\",\"success_count\":1,\"total_count\":1}\\n' "
        "\"${official_dir}\" > \"${official_dir}/native_eval_summary.json\"\n",
        encoding="utf-8",
    )
    fake_launcher.chmod(0o755)
    run_root = _initialize(
        tmp_path,
        selected=["0"],
        command_template=f"{fake_launcher} {{output_dir}}",
        worker_env=["CUDA_VISIBLE_DEVICES=1"],
    )

    assert manager.main(
        [
            "worker",
            "--run-root",
            str(run_root),
            "--worker-id",
            "fake-worker",
            "--worker-slot",
            "4",
            "--cuda-visible-devices",
            "3",
            "--max-episodes",
            "1",
        ]
    ) == 0

    stdout_path = next((run_root / "episodes").rglob("stdout.log"))
    environment_values = stdout_path.read_text(encoding="utf-8").strip().split("|")
    assert environment_values[:6] == [
        "10",
        "vase",
        "0",
        "http://127.0.0.1:11605",
        "unset",
        "false",
    ]
    assert environment_values[6].endswith("/debug")
    assert environment_values[7].endswith("/tmp")
    assert environment_values[8] == "unset"
    assert environment_values[9] == "3"
    claim_path = next((run_root / "episodes").rglob("claim.json"))
    claim_payload = json.loads(claim_path.read_text(encoding="utf-8"))
    assert claim_payload["claim"]["house_index"] == 10
    assert claim_payload["claim"]["target_type"] == "vase"
    assert claim_payload["environment_overrides"]["ROS_HOUSE_IND"] == "10"
    assert claim_payload["environment_overrides"]["ROS_TARGET_TYPES"] == "vase"
    assert claim_payload["environment_overrides"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert claim_payload["runtime_provenance"] == {
        "cuda_visible_devices": "3",
        "cuda_visible_devices_source": "worker_cli",
    }

    _root, config, ledger = manager._load_run_config(run_root)
    assert config["worker_environment"]["CUDA_VISIBLE_DEVICES"] == "1"
    status = ledger.status()
    assert status["counts"]["completed"] == 1
    assert status["active_worker_slot_count"] == 0


def test_run_maps_each_cuda_binding_to_one_spawned_worker(tmp_path: Path) -> None:
    fake_launcher = tmp_path / "fake_cuda_launcher.sh"
    fake_launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "output_dir=$1\n"
        "official_dir=${output_dir}/NativeNavToObjEvalConfig/fake\n"
        "mkdir -p ${official_dir}\n"
        "printf '%s\\n' \"${CUDA_VISIBLE_DEVICES-unset}\"\n"
        "printf '{\"official_eval_output_dir\":\"%s\",\"success_count\":1,\"total_count\":1}\\n' "
        "\"${official_dir}\" > \"${official_dir}/native_eval_summary.json\"\n",
        encoding="utf-8",
    )
    fake_launcher.chmod(0o755)
    run_root = _initialize(
        tmp_path,
        selected=["0", "1"],
        command_template=f"{fake_launcher} {{output_dir}}",
    )

    assert manager.main(
        [
            "run",
            "--run-root",
            str(run_root),
            "--workers",
            "2",
            "--max-episodes-per-worker",
            "1",
            "--cuda-visible-devices-list",
            "2,7",
        ]
    ) == 0

    claims = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run_root / "episodes").rglob("claim.json")
    ]
    assert len(claims) == 2
    assert {claim["environment_overrides"]["CUDA_VISIBLE_DEVICES"] for claim in claims} == {
        "2",
        "7",
    }
    assert {
        claim["runtime_provenance"]["cuda_visible_devices_source"] for claim in claims
    } == {"worker_cli"}
    assert {
        path.read_text(encoding="utf-8").strip()
        for path in (run_root / "episodes").rglob("stdout.log")
    } == {"2", "7"}


def test_run_staggers_worker_start_waves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = _initialize(tmp_path, selected=["0", "1", "2"])
    commands: list[list[str]] = []
    sleeps: list[float] = []

    class FakeProcess:
        def wait(self) -> int:
            return 0

    def fake_popen(command: list[str], *, start_new_session: bool) -> FakeProcess:
        assert start_new_session is True
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr(manager.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(manager.time, "sleep", sleeps.append)

    assert manager.main(
        [
            "run",
            "--run-root",
            str(run_root),
            "--workers",
            "3",
            "--worker-start-wave-size",
            "2",
            "--worker-start-interval-seconds",
            "0.25",
            "--cuda-visible-devices-list",
            "0,1,2",
        ]
    ) == 0

    assert [command[command.index("--worker-slot") + 1] for command in commands] == ["0", "1", "2"]
    assert sleeps == [0.25]


def test_cuda_binding_list_requires_exact_worker_count() -> None:
    assert manager._parse_cuda_visible_devices_list("0,1", workers=2) == ["0", "1"]
    with pytest.raises(ValueError, match="exactly one comma-separated binding"):
        manager._parse_cuda_visible_devices_list("0", workers=2)
