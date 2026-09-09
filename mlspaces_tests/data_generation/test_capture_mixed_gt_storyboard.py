from __future__ import annotations

import numpy as np
import pytest

from scripts.InteractiveNav import capture_mixed_gt_storyboard as storyboard


def representative_episode(
    *,
    target_category: str,
    crossed_roots: list[str] | None = None,
) -> dict:
    return {
        "task": {
            "robot_base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        },
        "interactive_nav": {
            "schema_version": "interactive_nav_v3",
            "interaction_domains": ["channel", "container"],
            "target": {
                "category": target_category,
                "container_category": "Fridge",
            },
            "initial_state": {
                "required_door_roots_closed": ["door_root"],
            },
            "interactions": [
                {"type": "channel_hinged_door"},
                {"type": "container_hinged_door"},
            ],
            "generation_validation": {
                "minimal_plan_verified": True,
                "navigation_validation": {
                    "all_open_path_crossed_door_roots": crossed_roots
                    if crossed_roots is not None
                    else ["door_root"],
                    "all_open_path_length_m": 6.0,
                    "approach_path_length_m": 2.5,
                    "initial_state_path_found": False,
                    "oracle_restored_path_found": True,
                    "interaction_pose_collision_free": True,
                    "door_approach": {"approach_xy": [1.0, 0.0]},
                    "interaction_pose": [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                },
            },
        },
    }


def test_choose_episode_prefers_single_door_apple_fridge(monkeypatch) -> None:
    episodes = [
        representative_episode(target_category="potato"),
        representative_episode(target_category="apple"),
        representative_episode(
            target_category="apple", crossed_roots=["door_root", "door_root_2"]
        ),
    ]
    monkeypatch.setattr(
        storyboard.interactive_nav_v3,
        "validate_mixed_v3_episode",
        lambda episode: None,
    )

    index, selected, metadata = storyboard.choose_episode(
        episodes,
        episode_index=None,
        case_id=None,
    )

    assert index == 1
    assert selected is episodes[1]
    assert metadata["selection_mode"] == "automatic_representative"
    assert metadata["candidate_count"] == 2


def test_build_story_steps_has_expected_five_state_sequence() -> None:
    episode = representative_episode(target_category="apple")
    steps = storyboard.build_story_steps(
        episode,
        approach_path=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        door_center_xy=np.asarray([2.0, 0.0]),
    )

    assert [step.name for step in steps] == [
        "start",
        "door_front_closed",
        "door_front_open",
        "fridge_front_closed",
        "fridge_front_open",
    ]
    assert [(step.door_state, step.container_state) for step in steps] == [
        ("closed", "closed"),
        ("closed", "closed"),
        ("open", "closed"),
        ("open", "closed"),
        ("open", "open"),
    ]
    assert steps[1].robot_pose == steps[2].robot_pose
    assert steps[3].robot_pose == steps[4].robot_pose


def test_shoulder_camera_is_right_rear_and_above_for_positive_x_heading() -> None:
    robot_pose = np.asarray([1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    config = storyboard.ShoulderCameraConfig()

    pose = storyboard.shoulder_camera_pose(robot_pose, config)

    assert pose["position"] == pytest.approx([0.28, 1.66, 1.72])
    assert pose["target"] == pytest.approx([2.75, 2.0, 1.08])
