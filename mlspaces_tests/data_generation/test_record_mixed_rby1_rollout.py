from __future__ import annotations

import argparse

import numpy as np
import pytest

from scripts.InteractiveNav import container_scene_probe as probe
from scripts.InteractiveNav import record_mixed_rby1_rollout as rollout


def test_compress_path_preserves_endpoints_and_spacing() -> None:
    path = np.asarray([[x / 10.0, 0.0] for x in range(11)], dtype=float)

    compressed = rollout.compress_path(path, spacing_m=0.25)

    assert compressed[0].tolist() == [0.0, 0.0]
    assert compressed[-1].tolist() == [1.0, 0.0]
    assert len(compressed) < len(path)
    assert np.linalg.norm(compressed[1] - compressed[0]) >= 0.25


def test_pose_for_xy_faces_the_next_pathpoint() -> None:
    pose = rollout.pose_for_xy(np.asarray([1.0, 2.0]), np.asarray([1.0, 3.0]))

    assert pose[:3, 3] == pytest.approx([1.0, 2.0, 0.0])
    assert pose[:2, 0] == pytest.approx([0.0, 1.0])


def test_frame_collector_keeps_segment_and_camera_indices() -> None:
    collector = rollout.FrameCollector()
    callback = collector.callback("door")
    frame = np.zeros((4, 5, 3), dtype=np.uint8)

    callback("head_camera", frame, {"phase": "OPEN_DOOR", "task_step": 12})

    assert len(collector.frames["head_camera"]) == 1
    assert collector.events == [
        {
            "segment": "door",
            "camera": "head_camera",
            "phase": "OPEN_DOOR",
            "task_step": 12,
            "frame_index": 0,
        }
    ]


def test_request_args_forwards_container_retry_controls(tmp_path) -> None:
    args = argparse.Namespace(
        door_arm="auto",
        container_arm="left",
        approach_distance=0.5,
        min_base_clearance=0.15,
        max_approach_distance=1.2,
        max_base_adjustment_distance=0.75,
        max_base_adjustment_steps=300,
        door_max_steps_per_waypoint=42,
        door_max_planning_reattempts=9,
        door_joint_position_tolerance=0.12,
        door_articulation_delta_deg=11.0,
        allow_force_fallback=False,
        force_fallback_target_fraction=1.0,
        force_fallback_max_steps=1500,
        success_threshold=0.67,
        max_steps=500,
        container_max_steps_per_waypoint=80,
        container_max_batch_plan_attempts=16,
        container_max_planning_reattempts=9,
        variant="base",
        seed=0,
        output_dir=tmp_path,
        data_split="val",
    )

    request = rollout.request_args(
        house_index=102,
        interaction_kind="container",
        target_name="fridge",
        joint_index=5,
        args=args,
    )

    assert request.container_arm == "left"
    assert request.container_max_steps_per_waypoint == 80
    assert request.container_max_batch_plan_attempts == 16
    assert request.container_max_planning_reattempts == 9
    assert request.door_max_steps_per_waypoint == 42
    assert request.door_max_planning_reattempts == 9
    assert request.door_joint_position_tolerance == 0.12
    assert request.door_articulation_delta_deg == 11.0


def test_semantic_open_fraction_supports_positive_and_negative_ranges() -> None:
    assert probe.semantic_open_fraction(0.75, 0.0, 1.5) == pytest.approx(0.5)
    assert probe.semantic_open_fraction(-0.75, 0.0, -1.5) == pytest.approx(0.5)
