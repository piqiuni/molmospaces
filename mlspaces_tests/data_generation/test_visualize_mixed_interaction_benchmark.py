from __future__ import annotations

import json

import pytest

from scripts.InteractiveNav import visualize_mixed_interaction_benchmark as visualize


def mixed_episode() -> dict:
    return {
        "house_index": 12,
        "task": {
            "robot_base_pose": [1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        },
        "interactive_nav": {
            "schema_version": "interactive_nav_v3",
            "case_id": "mixed_case_12",
            "interaction_domains": ["channel", "container"],
            "target": {
                "selected_instance": "target_object",
                "category": "apple",
                "object_aabb_center": [5.0, 6.0, 0.8],
                "container_name": "fridge",
                "container_category": "Fridge",
                "container_aabb_center": [5.0, 5.5, 1.0],
            },
            "initial_state": {
                "required_door_roots_closed": ["door_root"],
            },
            "interactions": [
                {
                    "type": "channel_hinged_door",
                    "object_name": "door_leaf",
                    "object_category": "Door",
                    "joint_index": 0,
                },
                {
                    "type": "container_hinged_door",
                    "object_name": "fridge",
                    "object_category": "Fridge",
                    "joint_index": 2,
                },
            ],
            "generation_validation": {
                "navigation_validation": {
                    "interaction_pose": [4.8, 5.4, 0.0, 1.0, 0.0, 0.0, 0.0],
                    "door_approach": {"approach_xy": [3.0, 4.0]},
                    "all_open_path_crossed_door_roots": ["door_root", "other_door"],
                    "all_open_path_length_m": 7.5,
                    "approach_path_length_m": 3.0,
                    "initial_state_path_found": False,
                }
            },
        },
    }


def plot_record(name: str, category: str = "Object") -> dict:
    return {
        "name": name,
        "category": category,
        "aabb_center": [1.0, 2.0, 0.5],
        "aabb_size": [0.4, 0.5, 1.0],
        "position": [1.0, 2.0, 0.5],
        "is_structural": False,
        "is_receptacle": False,
        "is_pickup_candidate": False,
        "is_articulable": False,
    }


def test_load_benchmark_episodes_accepts_list_and_wrapped_payload(tmp_path):
    episode = mixed_episode()
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps([episode]))
    wrapped_dir = tmp_path / "wrapped"
    wrapped_dir.mkdir()
    (wrapped_dir / "benchmark.json").write_text(json.dumps({"episodes": [episode]}))

    assert visualize.load_benchmark_episodes(list_path) == [episode]
    assert visualize.load_benchmark_episodes(wrapped_dir) == [episode]


def test_extract_episode_annotations_preserves_mixed_gt_fields():
    annotations = visualize.extract_episode_annotations(mixed_episode())

    assert annotations["case_id"] == "mixed_case_12"
    assert annotations["required_door_roots"] == ["door_root"]
    assert annotations["channel_object_names"] == ["door_leaf"]
    assert annotations["crossed_door_roots"] == ["door_root", "other_door"]
    assert annotations["start_xy"].tolist() == [1.0, 2.0]
    assert annotations["interaction_xy"].tolist() == [4.8, 5.4]
    assert annotations["recorded_gt_path_length_m"] == 7.5


def test_extract_episode_annotations_rejects_non_mixed_episode():
    episode = mixed_episode()
    episode["interactive_nav"]["interaction_domains"] = ["container"]

    with pytest.raises(ValueError, match="requires a mixed episode"):
        visualize.extract_episode_annotations(episode)


def test_object_catalog_assigns_roles_and_stable_plot_ids():
    annotations = visualize.extract_episode_annotations(mixed_episode())
    records = [
        plot_record("target_object", "Apple"),
        plot_record("fridge", "Fridge"),
        plot_record("door_root", "Door"),
        plot_record("chair", "Chair"),
    ]

    plot_ids, catalog = visualize.build_object_catalog(records, records, annotations)
    by_name = {row["name"]: row for row in catalog}

    assert plot_ids == {
        "chair": "O001",
        "door_root": "O002",
        "fridge": "O003",
        "target_object": "O004",
    }
    assert by_name["target_object"]["role"] == "target"
    assert by_name["fridge"]["role"] == "container"
    assert by_name["door_root"]["role"] == "required_channel_root"
    assert by_name["chair"]["role"] == "scene_object"


def test_compare_path_length_reports_tolerance_result():
    passed = visualize.compare_path_length(7.5, 7.7, 0.25)
    failed = visualize.compare_path_length(7.5, 7.8, 0.25)
    relatively_close = visualize.compare_path_length(12.0, 11.05, 0.35, 0.10)

    assert passed["passed"] is True
    assert passed["absolute_error_m"] == pytest.approx(0.2)
    assert failed["passed"] is False
    assert relatively_close["passed"] is True
    assert relatively_close["allowed_error_m"] == pytest.approx(1.2)
