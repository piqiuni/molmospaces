"""Offline contract tests for the mixed-domain benchmark scheduler."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.InteractiveNav.evaluation.benchmark_io import load_benchmark_episodes
from scripts.InteractiveNav.evaluation.benchmark_policies import build_factory_policy
from scripts.InteractiveNav.evaluation.benchmark_types import PolicyObservation, PublicEpisode
from scripts.InteractiveNav.evaluation.episode_topdown import _load_benchmark_episode


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "InteractiveNav"
    / "run_interactive_nav_benchmark_eval.py"
)
REPO_ROOT = SCRIPT.parents[2]
CANONICAL_SCRIPT = (
    REPO_ROOT / "scripts" / "InteractiveNav" / "evaluate_interactive_nav_v3.py"
)
SPEC = importlib.util.spec_from_file_location("interactive_nav_mixed_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_gzip_json(path: Path, payload: object) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)


def test_selector_round_robins_three_domains() -> None:
    selected = MODULE.select_domain_episodes(
        {
            "channel": [{"id": index} for index in range(5)],
            "container": [{"id": index} for index in range(5)],
            "mixed": [{"id": index} for index in range(5)],
        },
        max_episodes=7,
    )

    assert [(item.domain, item.source_index) for item in selected] == [
        ("channel", 0),
        ("container", 0),
        ("mixed", 0),
        ("channel", 1),
        ("container", 1),
        ("mixed", 1),
        ("channel", 2),
    ]


def test_selector_caps_each_domain_and_preserves_local_indices() -> None:
    selected = MODULE.select_domain_episodes(
        {
            "channel": [{"id": index} for index in range(2)],
            "container": [{"id": index} for index in range(4)],
            "mixed": [{"id": index} for index in range(3)],
        },
        episodes_per_domain=3,
    )

    assert [(item.domain, item.source_index) for item in selected] == [
        ("channel", 0),
        ("container", 0),
        ("mixed", 0),
        ("channel", 1),
        ("container", 1),
        ("mixed", 1),
        ("container", 2),
        ("mixed", 2),
    ]


def test_progress_formal_sr_uses_result_success_and_eligibility() -> None:
    rows = [
        {"success": True, "nav_success": False, "scoring_eligible": True},
        {"success": False, "nav_success": True, "scoring_eligible": True},
        {"success": True, "nav_success": True, "scoring_eligible": False},
    ]

    assert MODULE._episode_sr(rows) == 0.5
    assert MODULE._episode_nav_sr(rows) == 0.5


def test_gzip_benchmark_loader_and_dry_run(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmarks"
    benchmark_root.mkdir()
    for domain in MODULE.DOMAIN_NAMES:
        _write_gzip_json(benchmark_root / f"{domain}.json.gz", [{"id": domain}])

    resolved, episodes = load_benchmark_episodes(benchmark_root / "channel.json.gz")
    assert resolved.name == "channel.json.gz"
    assert episodes == [{"id": "channel"}]
    assert _load_benchmark_episode(
        benchmark_root / "channel.json.gz", episode_index=0, case_id=None
    ) == {"id": "channel"}

    output_dir = tmp_path / "eval"
    assert (
        MODULE.main(
            [
                "--output-dir",
                str(output_dir),
                "--benchmark-root",
                str(benchmark_root),
                "--policy",
                "factory",
                "--policy-factory",
                "scripts.InteractiveNav.evaluation.example_external_policy:build_policy",
                "--policy-kwargs-json",
                '{"reason":"manifest_contract"}',
                "--max-steps",
                "17",
                "--min-steps",
                "11",
                "--camera-names",
                "head_camera",
                "wrist_camera",
                "--image-resolution",
                "320",
                "240",
                "--video-fps",
                "7.5",
                "--dry-run",
            ]
        )
        == 0
    )
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_episode_count"] == 3
    assert manifest["domain_counts"] == {"channel": 1, "container": 1, "mixed": 1}
    for domain in MODULE.DOMAIN_NAMES:
        payload = manifest["domains"][domain]["run_signature_payload"]
        assert payload["protocol_version"]
        assert payload["protocol_implementation_sha256"]
        assert payload["paper_metric_config"]
        config = payload["evaluation_config"]
        assert config["policy_kwargs"] == {"reason": "manifest_contract"}
        assert config["max_steps"] == 17
        assert config["min_steps"] == 11
        assert config["camera_names"] == ["head_camera", "wrist_camera"]
        assert config["image_resolution"] == [320, 240]
        assert config["video_fps"] == 7.5


def test_canonical_cli_imports_without_ros_pythonpath(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "TMPDIR": str(runtime_root),
            "XDG_CACHE_HOME": str(runtime_root / "xdg"),
            "HF_HOME": str(runtime_root / "hf"),
            "TORCH_HOME": str(runtime_root / "torch"),
        }
    )

    result = subprocess.run(
        [sys.executable, str(CANONICAL_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--policy-factory" in result.stdout


def test_example_external_factory_uses_public_contract() -> None:
    public = PublicEpisode(
        house_index=19,
        scene_dataset="procthor-10k",
        data_split="val",
        instruction="Find the alarm clock.",
        task_type="nav_to_obj",
        camera_names=["head_camera"],
        image_resolution=(640, 480),
    )
    policy = build_factory_policy(
        "scripts.InteractiveNav.evaluation.example_external_policy:build_policy",
        public_episode=public,
        kwargs={"reason": "contract_test"},
    )
    policy.reset(public)
    action = policy.act(
        PolicyObservation(
            observation={"head_camera": "public-sensor-payload"},
            instruction=public.instruction,
            step_index=0,
            elapsed_seconds=0.0,
            previous_action=None,
        )
    )

    assert action.kind == "stop"
    assert action.metadata["wrapped_action"]["reason"] == "contract_test"
    assert set(policy.policy.public_episode) == {
        "house_index",
        "scene_dataset",
        "data_split",
        "instruction",
        "task_type",
        "camera_names",
        "image_resolution",
    }


def test_bundled_benchmark_matches_manifest() -> None:
    root = MODULE.DEFAULT_BENCHMARK_ROOT
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    total = 0
    case_ids: set[str] = set()
    for domain in MODULE.DOMAIN_NAMES:
        expected = manifest["domains"][domain]
        archive = root / expected["file"]
        assert archive.stat().st_size == expected["archive_size_bytes"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected["archive_sha256"]

        digest = hashlib.sha256()
        with gzip.open(archive, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == expected["uncompressed_sha256"]

        _resolved, episodes = load_benchmark_episodes(archive)
        assert len(episodes) == expected["episode_count"]
        for episode in episodes:
            case_id = str(episode["interactive_nav"]["case_id"])
            assert case_id not in case_ids
            case_ids.add(case_id)
        total += len(episodes)
    assert total == manifest["formal_episode_count"]
