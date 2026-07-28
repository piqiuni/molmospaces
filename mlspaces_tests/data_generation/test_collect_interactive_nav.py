from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.InteractiveNav import collect_interactive_nav as collector
from scripts.InteractiveNav import build_mixed_interaction_benchmark as mixed_builder
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
from scripts.InteractiveNav.collection.progress_reporter import _active_batch, snapshot
from scripts.InteractiveNav.collection.scene_source import build_scene_manifest
from scripts.InteractiveNav.collection.seed_builder import (
    _candidate_start_pose,
    load_nav_benchmark_source,
)


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
    interactions = []
    if domain in {"channel", "mixed"}:
        interactions.append({"type": "channel_hinged_door"})
    if domain in {"container", "mixed"}:
        interactions.append({"type": "container_hinged_door"})
    return {
        "house_index": house,
        "interactive_nav": {
            "case_id": case_id,
            "interaction_domains": domains,
            "interaction_requirement": (
                "unnecessary" if recipe == "distractor_doors_closed" else "required"
            ),
            "interactions": interactions,
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


def test_source_config_exposes_scene_preferences(tmp_path: Path) -> None:
    payload = config_payload(tmp_path)
    payload["source"].update(
        {
            "kind": "scene_split",
            "preferred_object_categories": ["book", "cellphone"],
            "preferred_object_names": ["book_instance_1"],
            "min_start_goal_distance_m": 3.0,
            "max_start_goal_distance_m": 12.0,
            "prefer_longest_start_goal": False,
        }
    )

    config = CollectionConfig.model_validate(payload)

    assert config.source.kind == "scene_split"
    assert config.source.preferred_object_categories == ["book", "cellphone"]
    assert config.source.min_start_goal_distance_m == 3.0
    assert config.source.max_start_goal_distance_m == 12.0
    assert config.source.prefer_longest_start_goal is False


def test_strict_three_way_3000_config_has_1000_per_domain(tmp_path: Path) -> None:
    payload = config_payload(tmp_path, total_samples=3000)
    payload["balance"].update(
        {
            "enforce_three_way_balance": True,
            "domain_order": ["channel", "container", "mixed"],
            "balance_target_categories": "equal",
            "balance_path_lengths": "equal",
        }
    )
    config = CollectionConfig.model_validate(payload)
    assert config.balance.target_counts() == {
        "channel": 1000,
        "container": 1000,
        "mixed": 1000,
    }


def test_strict_three_way_rejects_non_divisible_total(tmp_path: Path) -> None:
    payload = config_payload(tmp_path, total_samples=10)
    payload["balance"]["enforce_three_way_balance"] = True
    with pytest.raises(ValueError, match="divisible by 3"):
        CollectionConfig.model_validate(payload)


def test_seed_resume_fingerprint_changes_with_seed_capacity(tmp_path: Path) -> None:
    first = CollectionConfig.model_validate(config_payload(tmp_path))
    payload = config_payload(tmp_path)
    payload["source"]["seeds_per_house"] = first.source.seeds_per_house + 1
    second = CollectionConfig.model_validate(payload)
    assert collector._seed_generation_config(first) != collector._seed_generation_config(
        second
    )


def test_collection_fingerprint_ignores_execution_parallelism_but_not_seed(
    tmp_path: Path,
) -> None:
    seed_benchmark = tmp_path / "benchmark.json"
    container_rough = tmp_path / "rough_catalog.json"
    mixed_rough = tmp_path / "mixed_rough_catalog.json"
    write_json(seed_benchmark, [{"house_index": 1}])
    write_json(container_rough, {"schema_version": "container_rough_catalog_v1"})
    write_json(mixed_rough, {"schema_version": "mixed_rough_catalog_v1"})
    payload = config_payload(tmp_path)
    payload["runtime"] = {
        "workers": 1,
        "light_scheduler": "sequential",
        "seed": 7,
    }
    single = CollectionConfig.model_validate(payload)
    payload["runtime"] = {
        "workers": 4,
        "light_scheduler": "domain_parallel",
        "seed": 7,
    }
    parallel = CollectionConfig.model_validate(payload)
    payload["runtime"] = {"workers": 4, "seed": 8}
    changed_seed = CollectionConfig.model_validate(payload)
    payload = config_payload(tmp_path)
    payload["runtime"] = {
        "workers": 4,
        "light_scheduler": "domain_parallel",
        "seed": 7,
    }
    payload["balance"]["enforce_max_samples_per_house"] = False
    unrestricted_balance = CollectionConfig.model_validate(payload)

    rough_paths = [container_rough, mixed_rough]
    single_fingerprint = collector._collection_fingerprint(
        single,
        seed_benchmark=seed_benchmark,
        rough_paths=rough_paths,
    )
    parallel_fingerprint = collector._collection_fingerprint(
        parallel,
        seed_benchmark=seed_benchmark,
        rough_paths=rough_paths,
    )
    changed_seed_fingerprint = collector._collection_fingerprint(
        changed_seed,
        seed_benchmark=seed_benchmark,
        rough_paths=rough_paths,
    )
    unrestricted_balance_fingerprint = collector._collection_fingerprint(
        unrestricted_balance,
        seed_benchmark=seed_benchmark,
        rough_paths=rough_paths,
    )

    assert single_fingerprint == parallel_fingerprint
    assert single_fingerprint == unrestricted_balance_fingerprint
    assert single_fingerprint != changed_seed_fingerprint


def test_mixed_candidate_seed_is_key_stable() -> None:
    candidate = {
        "case_id": "mixed-case",
        "_preferred_source_episode_index": 17,
    }
    assert mixed_builder.candidate_seed(123, candidate) == mixed_builder.candidate_seed(
        123, dict(candidate)
    )
    assert mixed_builder.candidate_seed(123, candidate) != mixed_builder.candidate_seed(
        123,
        {**candidate, "_preferred_source_episode_index": 18},
    )


def test_scene_start_sampler_enforces_straight_line_distance_preferences() -> None:
    class FakeSceneMap:
        @staticmethod
        def pos_m_to_px(values):
            array = np.asarray(values)
            return np.rint(array[..., :2]).astype(int)

    class FakeEnv:
        current_robot = SimpleNamespace(
            robot_view=SimpleNamespace(base=SimpleNamespace(pose=np.eye(4)))
        )

        @staticmethod
        def check_if_robot_collision_at_base_pose(_robot_view, _pose):
            return False

    pose = _candidate_start_pose(
        SimpleNamespace(env=FakeEnv()),
        FakeSceneMap(),
        np.asarray([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
        np.asarray([0.0, 0.0]),
        np.random.default_rng(0),
        candidate_pool=3,
        component_labels=np.ones((10, 10), dtype=int),
        min_distance_m=3.0,
        max_distance_m=5.0,
        prefer_longest=True,
    )

    assert pose is not None
    assert np.allclose(pose[:2, 3], [4.0, 0.0])


def test_nav_benchmark_source_preserves_original_target_and_start(tmp_path: Path) -> None:
    benchmark = tmp_path / "nav_benchmark.json"
    episodes = [
        {
            "house_index": 3,
            "scene_dataset": "procthor-10k",
            "data_split": "val",
            "seed": 11,
            "task": {
                "pickup_obj_name": "book_1",
                "robot_base_pose": [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            },
        },
        {
            "house_index": 4,
            "scene_dataset": "procthor-10k",
            "data_split": "train",
            "seed": 12,
            "task": {
                "pickup_obj_name": "lamp_1",
                "robot_base_pose": [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            },
        },
    ]
    write_json(benchmark, episodes)
    payload = config_payload(tmp_path)
    payload["source"].update(
        {
            "kind": "nav_benchmark",
            "benchmark_path": str(benchmark),
            "data_split": "val",
            "houses": [3],
        }
    )
    config = CollectionConfig.model_validate(payload)

    selected, manifest = load_nav_benchmark_source(config.source)

    assert len(selected) == 1
    assert selected[0]["task"]["pickup_obj_name"] == "book_1"
    assert selected[0]["task"]["robot_base_pose"] == episodes[0]["task"]["robot_base_pose"]
    assert selected[0]["seed_generation"]["start_goal_policy"] == (
        "preserve_original_nav_benchmark_episode"
    )
    assert manifest["selected_house_indices"] == [3]


def test_nav_benchmark_source_rejects_missing_split(tmp_path: Path) -> None:
    benchmark = tmp_path / "nav_benchmark.json"
    write_json(
        benchmark,
        [
            {
                "house_index": 0,
                "data_split": "val",
                "task": {"pickup_obj_name": "book", "robot_base_pose": []},
            }
        ],
    )
    payload = config_payload(tmp_path)
    payload["source"].update(
        {"kind": "nav_benchmark", "benchmark_path": str(benchmark), "data_split": "train"}
    )
    config = CollectionConfig.model_validate(payload)

    with pytest.raises(RuntimeError, match="No episodes with data_split"):
        load_nav_benchmark_source(config.source)


def test_build_seed_benchmark_uses_nav_source_without_resampling(tmp_path: Path) -> None:
    benchmark = tmp_path / "nav_benchmark.json"
    original = {
        "house_index": 2,
        "scene_dataset": "procthor-10k",
        "data_split": "val",
        "seed": 3,
        "task": {
            "pickup_obj_name": "target_1",
            "robot_base_pose": [1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
    }
    write_json(benchmark, [original])
    payload = config_payload(tmp_path)
    payload["source"].update(
        {
            "kind": "nav_benchmark",
            "benchmark_path": str(benchmark),
            "data_split": "val",
            "houses": "all",
        }
    )
    config = CollectionConfig.model_validate(payload)

    source_path = collector.build_seed_benchmark(config)
    selected = json.loads(source_path.read_text())

    assert source_path == tmp_path / "source" / "benchmark.json"
    assert selected[0]["task"] == original["task"]
    assert (tmp_path / "source" / "benchmark_manifest.json").exists()


def test_container_and_mixed_fine_require_rough_catalogs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = config_payload(tmp_path)
    payload["rough"] = {"generate_if_missing": False}
    config = CollectionConfig.model_validate(payload)
    seed_benchmark = tmp_path / "seeds" / "benchmark.json"
    write_json(seed_benchmark, [])
    monkeypatch.setattr(collector, "run_door_parallel", lambda *_args, **_kwargs: None)

    with pytest.raises(FileNotFoundError, match="precomputed rough catalog"):
        collector.run_light_collectors(config, seed_benchmark)


def test_domain_parallel_scheduler_launches_three_independent_queues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = config_payload(tmp_path, total_samples=9)
    payload["runtime"] = {"workers": 3, "light_scheduler": "domain_parallel"}
    payload["rough"] = {
        "container_catalog": str(tmp_path / "rough_catalog.json"),
        "mixed_catalog": str(tmp_path / "mixed_rough_catalog.json"),
        "generate_if_missing": False,
    }
    config = CollectionConfig.model_validate(payload)
    seed_benchmark = tmp_path / "seeds" / "benchmark.json"
    write_json(seed_benchmark, [{"house_index": 0}])
    write_json(
        config.rough.container_catalog,
        {"houses": [{"house_index": 0, "strict_pair_count": 1}]},
    )
    write_json(
        config.rough.mixed_catalog,
        {
            "candidates": [
                {
                    "house_index": 0,
                    "rough_candidate_type": "mixed_required_verified",
                    "mixed_required_verified": True,
                }
            ]
        },
    )
    monkeypatch.setattr(collector, "_ensure_collection_fingerprint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(collector, "_write_or_validate_house_plan", lambda *_args, **_kwargs: None)

    calls: list[str] = []

    def fake_queue(domain: str):
        def run(*_args, **_kwargs):
            calls.append(domain)
            path = tmp_path / "raw" / domain / "benchmark.json"
            recipe = "single_path_door_closed" if domain == "channel" else ""
            write_json(path, [fake_episode(domain, 0, f"{domain}-case", recipe)])
            return path

        return run

    monkeypatch.setattr(collector, "run_container_fine_collection", fake_queue("container"))
    monkeypatch.setattr(collector, "run_door_parallel", fake_queue("channel"))
    monkeypatch.setattr(collector, "run_mixed_fine_parallel", fake_queue("mixed"))

    paths = collector.run_light_collectors(
        config,
        seed_benchmark,
        targets={"channel": 1, "container": 1, "mixed": 1},
    )

    assert set(calls) == {"channel", "container", "mixed"}
    assert set(paths) == {"channel", "container", "mixed"}
    ledger = json.loads((tmp_path / "raw" / "domain_scheduler.ledger.json").read_text())
    assert ledger["workers_total"] == 3
    assert ledger["rounds"][0]["worker_allocation"] == {
        "channel": 1,
        "container": 1,
        "mixed": 1,
    }
    assert ledger["maximum_simulation_count"] == 3


def test_domain_worker_allocation_hands_released_slots_to_mixed() -> None:
    assert collector._domain_worker_allocation(
        3,
        ["container", "mixed"],
        current_counts={"container": 900, "mixed": 100},
        targets={"container": 1000, "mixed": 1000},
    ) == {"container": 1, "mixed": 2}
    assert collector._domain_worker_allocation(
        3,
        ["mixed"],
        current_counts={"mixed": 100},
        targets={"mixed": 1000},
    ) == {"mixed": 3}


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
        "action_type_counts": {"force_joint": 3},
        "segment_counts": {"container_open": 3},
        "terminal_step_count": 1,
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
    def fake_validate(path: Path) -> dict:
        text = str(path)
        if "channel-case" in text:
            segments = {
                "nav_to_door": 1,
                "force_open_door": 1,
                "nav_to_target": 1,
                "terminal_observation": 1,
            }
        elif "container-case" in text:
            segments = {
                "nav_to_container": 1,
                "force_open_container": 1,
                "terminal_observation": 1,
            }
        else:
            segments = {
                "nav_to_door": 1,
                "force_open_door": 1,
                "nav_to_container": 1,
                "force_open_container": 1,
                "terminal_observation": 1,
            }
        return {
            "success": True,
            "step_count": sum(segments.values()),
            "action_type_counts": {"force_joint": 2},
            "segment_counts": segments,
            "terminal_step_count": 1,
        }

    monkeypatch.setattr(collector, "validate_full_rollout", fake_validate)

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


def test_full_stage_retries_candidates_until_each_domain_has_a_valid_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = config_payload(tmp_path)
    payload["collection"]["mode"] = "full"
    payload["full"] = {
        "max_episodes": 1,
        "max_candidate_attempts_per_domain": 2,
        "domains": ["channel", "container", "mixed"],
    }
    config = CollectionConfig.model_validate(payload)
    benchmark = tmp_path / "balanced" / "benchmark.json"
    write_json(
        benchmark,
        [
            fake_episode("channel", 0, "channel-case-a"),
            fake_episode("channel", 1, "channel-case-b"),
            fake_episode("container", 0, "container-case"),
            fake_episode("mixed", 0, "mixed-case"),
        ],
    )

    def fake_run(command: list[str], *, log_path: Path, env: dict[str, str]) -> int:
        del log_path, env
        case_id = command[command.index("--case_id") + 1]
        if case_id == "channel-case-a":
            return 1
        output_dir = Path(command[command.index("--output_dir") + 1])
        run_dir = output_dir / f"run_{case_id[:48]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "trajectory.h5").touch()
        return 0

    def fake_validate(path: Path) -> dict:
        text = str(path)
        if "channel-case" in text:
            segments = {
                "nav_to_door": 1,
                "force_open_door": 1,
                "nav_to_target": 1,
                "terminal_observation": 1,
            }
        elif "container-case" in text:
            segments = {
                "nav_to_container": 1,
                "force_open_container": 1,
                "terminal_observation": 1,
            }
        else:
            segments = {
                "nav_to_door": 1,
                "force_open_door": 1,
                "nav_to_container": 1,
                "force_open_container": 1,
                "terminal_observation": 1,
            }
        return {
            "success": True,
            "action_type_counts": {"force_joint": 2},
            "segment_counts": segments,
            "terminal_step_count": 1,
        }

    monkeypatch.setattr(collector, "run_command", fake_run)
    monkeypatch.setattr(collector, "validate_full_rollout", fake_validate)

    summary = json.loads(collector.run_full_collectors(config).read_text())

    assert summary["attempted_trajectory_count"] == 4
    assert summary["valid_trajectory_count"] == 3
    assert summary["valid_trajectory_count_by_domain"] == {
        "channel": 1,
        "container": 1,
        "mixed": 1,
    }
    assert summary["runs"][0]["case_id"] == "channel-case-a"
    assert summary["runs"][0]["training_eligible"] is False
    assert summary["runs"][1]["case_id"] == "channel-case-b"
    assert summary["runs"][1]["training_eligible"] is True


def test_file_based_progress_snapshot_reports_container_eta_inputs(tmp_path: Path) -> None:
    batch = tmp_path / "raw" / "container_batches" / "batch_003"
    shard = batch / "shards" / "shard_000"
    write_json(
        shard / "collection_plan.json",
        {"selection": {"selected_house_count": 5}},
    )
    write_json(
        shard / "benchmark" / "house_catalog.partial.json",
        [{"house_index": 1}, {"house_index": 2}],
    )
    write_json(
        shard / "benchmark" / "benchmark.partial.json",
        [fake_episode("container", 1, "container-1")],
    )

    line = snapshot(
        tmp_path,
        global_target=3000,
        domain_targets={"channel": 300, "container": 300, "mixed": 300},
    )

    assert "stage=container" in line
    assert "batch=batch_003" in line
    assert "batch_progress=2/5" in line
    assert "batch_valid=1" in line
    assert "workers=1" in line
    assert "worker_loads=5" in line
    assert "domains=channel:0/300,container:1/300,mixed:0/300" in line
    assert "checkpoint_progress=1/900" in line
    assert "checkpoint_remaining=899" in line
    assert "eta=" in line
    assert "time=" in line
    assert "global_raw_valid=" in line
    assert "global_balanced=" in line
    assert "global_eta=" in line


def test_progress_reporter_ignores_stale_unfinalized_batch(tmp_path: Path) -> None:
    stale = tmp_path / "raw" / "container_batches" / "batch_006"
    stale.mkdir(parents=True)
    write_json(stale / "benchmark.partial.json", [])
    completed = tmp_path / "raw" / "container_batches" / "batch_008"
    write_json(completed / "batch_meta.json", {"batch_index": 8})

    assert _active_batch(tmp_path, "raw/container_batches/batch_*") is None
