from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.InteractiveNav import collect_interactive_nav as collector
from scripts.InteractiveNav import interactive_nav_v3
from scripts.InteractiveNav.collection.config import CollectionConfig
from scripts.InteractiveNav.collection.interaction_executors import (
    ForceInteractionExecutor,
    build_interaction_executor,
)
from scripts.InteractiveNav.collection.full_rollout_recorder import (
    H5StepRolloutRecorder,
    validate_full_rollout,
)
from scripts.InteractiveNav.collection.scene_source import build_scene_manifest


def config_payload(output_root: Path, *, total_samples: int = 10) -> dict:
    return {
        "source": {
            "scene_dataset": "procthor-10k",
            "data_split": "train",
            "houses": [0, 1],
            "variant": "base",
        },
        "collection": {
            "mode": "light",
            "open_gt_control": False,
            "synthetic_wrong_action_rollout": False,
        },
        "balance": {"total_samples": total_samples},
        "output": {"root": str(output_root)},
    }


def fake_episode(domain: str, house: int, case_id: str, recipe: str = "") -> dict:
    domains = ["channel", "container"] if domain == "mixed" else [domain]
    return {
        "house_index": house,
        "interactive_nav": {
            "case_id": case_id,
            "interaction_domains": domains,
            "interaction_requirement": (
                "unnecessary" if recipe == "distractor_doors_closed" else "required"
            ),
            "legacy_case_type": recipe,
        },
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_config_rejects_excluded_collection_modes(tmp_path: Path) -> None:
    payload = config_payload(tmp_path)
    payload["collection"]["open_gt_control"] = True
    with pytest.raises(ValueError, match="open_gt_control"):
        CollectionConfig.model_validate(payload)
    payload = config_payload(tmp_path)
    payload["collection"]["synthetic_wrong_action_rollout"] = True
    with pytest.raises(ValueError, match="wrong-action"):
        CollectionConfig.model_validate(payload)


def test_full_mode_cannot_silently_run_the_light_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = config_payload(tmp_path)
    payload["collection"]["mode"] = "full"
    config_path = tmp_path / "full.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(payload))
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect_interactive_nav.py",
            "--config",
            str(config_path),
            "--stage",
            "light",
        ],
    )

    with pytest.raises(ValueError, match="Use --stage full"):
        collector.main()


def test_full_config_accepts_all_three_domains(tmp_path: Path) -> None:
    payload = config_payload(tmp_path)
    payload["collection"]["mode"] = "full"
    payload["full"] = {
        "max_episodes": 1,
        "domains": ["channel", "container", "mixed"],
    }

    config = CollectionConfig.model_validate(payload)

    assert config.full.domains == ["channel", "container", "mixed"]


def test_train_scene_manifest_uses_requested_house_subset(tmp_path: Path) -> None:
    config = CollectionConfig.model_validate(config_payload(tmp_path))
    manifest = build_scene_manifest(config.source)

    assert manifest["data_split"] == "train"
    assert manifest["available_house_count"] == 2
    assert [row["house_index"] for row in manifest["houses"]] == [0, 1]
    assert all(row["asset_version"] == "20251122" for row in manifest["houses"])


def test_force_executor_is_registered() -> None:
    assert isinstance(build_interaction_executor("force"), ForceInteractionExecutor)
    with pytest.raises(ValueError, match="Unknown interaction executor"):
        build_interaction_executor("not-registered")


def test_balance_builds_4_3_3_without_all_open_or_all_closed(tmp_path: Path) -> None:
    config = CollectionConfig.model_validate(config_payload(tmp_path, total_samples=10))
    channel = [
        fake_episode("channel", 0, "required-0", "single_path_door_closed"),
        fake_episode("channel", 1, "required-1", "single_path_door_closed"),
        fake_episode("channel", 2, "distractor-0", "distractor_doors_closed"),
        fake_episode("channel", 3, "stress-0", "mixed_critical_and_distractor_closed"),
        fake_episode("channel", 4, "excluded-all-closed", "all_closed"),
    ]
    container = [fake_episode("container", index, f"container-{index}") for index in range(3)]
    mixed = [fake_episode("mixed", index, f"mixed-{index}") for index in range(3)]
    paths = {
        "channel": tmp_path / "raw/channel/benchmark.json",
        "container": tmp_path / "raw/container/benchmark.json",
        "mixed": tmp_path / "raw/mixed/benchmark.json",
    }
    write_json(paths["channel"], channel)
    write_json(paths["container"], container)
    write_json(paths["mixed"], mixed)

    benchmark_path = collector.balance_benchmark(config, paths)
    balanced = json.loads(benchmark_path.read_text())

    assert len(balanced) == 10
    counts = {"channel": 0, "container": 0, "mixed": 0}
    for episode in balanced:
        domains = episode["interactive_nav"]["interaction_domains"]
        key = "mixed" if len(domains) == 2 else domains[0]
        counts[key] += 1
        assert episode["interactive_nav"].get("legacy_case_type") != "all_closed"
    assert counts == {"channel": 4, "container": 3, "mixed": 3}


def test_channel_recipe_quota_is_soft_but_domain_count_is_strict(tmp_path: Path) -> None:
    config = CollectionConfig.model_validate(config_payload(tmp_path, total_samples=10))
    channel = [
        fake_episode("channel", index, f"required-{index}", "single_path_door_closed")
        for index in range(4)
    ]
    container = [fake_episode("container", index, f"container-{index}") for index in range(3)]
    mixed = [fake_episode("mixed", index, f"mixed-{index}") for index in range(3)]
    paths = {
        "channel": tmp_path / "raw/channel/benchmark.json",
        "container": tmp_path / "raw/container/benchmark.json",
        "mixed": tmp_path / "raw/mixed/benchmark.json",
    }
    write_json(paths["channel"], channel)
    write_json(paths["container"], container)
    write_json(paths["mixed"], mixed)

    balanced = json.loads(collector.balance_benchmark(config, paths).read_text())

    selected_channel = [
        episode
        for episode in balanced
        if episode["interactive_nav"]["interaction_domains"] == ["channel"]
    ]
    assert len(selected_channel) == 4
    assert {
        episode["interactive_nav"]["legacy_case_type"]
        for episode in selected_channel
    } == {"single_path_door_closed"}


def test_door_collection_resumes_from_incremental_samples(tmp_path: Path) -> None:
    config = CollectionConfig.model_validate(config_payload(tmp_path, total_samples=10))
    raw_root = tmp_path / "raw"
    for index in range(4):
        write_json(
            raw_root
            / "channel_shards"
            / "shard_000"
            / "output"
            / "samples"
            / f"sample_{index}"
            / "sample.json",
            fake_episode("channel", index, f"channel-{index}", "single_path_door_closed"),
        )

    benchmark_path = collector.run_door_parallel(config, [], raw_root, {})

    assert len(json.loads(benchmark_path.read_text())) == 4
    summary = json.loads((raw_root / "channel" / "summary.json").read_text())
    assert summary["resumed_from_incremental_samples"] is True
    assert summary["worker_count"] == 0


def test_v3_serializer_preserves_required_nullable_oracle_prefix_fields() -> None:
    example_path = (
        Path(interactive_nav_v3.__file__).parent
        / "dataset_definition/v3/examples/channel_episode.json"
    )
    example = json.loads(example_path.read_text())
    episode = collector.load_template_episode()
    episode["task"] = example["task"]
    episode["language"] = example["language"]
    episode["interactive_nav"] = example["interactive_nav"]
    prefix = episode["interactive_nav"]["generation_validation"]["oracle_prefixes"][0]
    for key in [
        "robot_reachable_to_next_goal",
        "target_distance_passed",
        "target_visibility_fraction",
        "task_success",
    ]:
        prefix[key] = None

    validated = interactive_nav_v3._serialize_and_validate_v3(episode)

    serialized_prefix = validated["interactive_nav"]["generation_validation"][
        "oracle_prefixes"
    ][0]
    assert all(key in serialized_prefix for key in prefix if key != "target_visible_pixels")
    assert serialized_prefix["robot_reachable_to_next_goal"] is None


def test_full_rollout_recorder_aligns_images_actions_and_state(tmp_path: Path) -> None:
    path = tmp_path / "rollout.h5"
    recorder = H5StepRolloutRecorder(
        path,
        episode_id="mixed-case-1",
        camera_names=["head_camera", "camera_follower"],
        metadata={"domain": "mixed", "policy": "force"},
    )
    for step in range(3):
        recorder.record_step(
            images={
                "head_camera": np.full((8, 12, 3), step, dtype=np.uint8),
                "camera_follower": np.full((8, 12, 3), step + 1, dtype=np.uint8),
            },
            action={"type": "force_joint", "vector": [step, 1.0]},
            state={"qpos": [step, step + 0.5], "qvel": [0.1, 0.2]},
            segment="container_open",
            phase="FORCE_OPEN",
            terminal=step == 2,
        )
    recorder.finalize(success=True, terminal_reason="completed")

    audit = validate_full_rollout(path)

    assert audit == {
        "schema_version": "interactive_nav_full_rollout_v1",
        "episode_id": "mixed-case-1",
        "step_count": 3,
        "success": True,
        "terminal_reason": "completed",
        "camera_names": ["camera_follower", "head_camera"],
    }


def test_full_stage_dispatches_all_domains_and_requires_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = config_payload(tmp_path)
    payload["collection"]["mode"] = "full"
    payload["full"] = {
        "max_episodes": 1,
        "domains": ["channel", "container", "mixed"],
    }
    config = CollectionConfig.model_validate(payload)
    benchmark = tmp_path / "balanced" / "benchmark.json"
    write_json(
        benchmark,
        [
            fake_episode("channel", 0, "channel-case"),
            fake_episode("container", 0, "container-case"),
            fake_episode("mixed", 0, "mixed-case"),
        ],
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, log_path: Path, env: dict[str, str]) -> int:
        del log_path, env
        commands.append(command)
        case_id = command[command.index("--case_id") + 1]
        output_dir = Path(command[command.index("--output_dir") + 1])
        run_dir = output_dir / f"run_{case_id[:48]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "trajectory.h5").touch()
        return 0

    monkeypatch.setattr(collector, "run_command", fake_run)
    monkeypatch.setattr(
        collector,
        "validate_full_rollout",
        lambda _path: {"success": True, "step_count": 3},
    )

    summary_path = collector.run_full_collectors(config)
    summary = json.loads(summary_path.read_text())

    assert summary["requested_episode_count"] == 3
    assert summary["valid_trajectory_count"] == 3
    assert {row["domain"] for row in summary["runs"]} == {
        "channel",
        "container",
        "mixed",
    }
    assert all(row["training_eligible"] for row in summary["runs"])
    command_text = [" ".join(command) for command in commands]
    assert any("record_interactive_nav_rby1_rollout.py" in command for command in command_text)
    assert any("record_mixed_rby1_rollout.py" in command for command in command_text)
