"""Protocol-only tests for the evaluator-owned restricted GT publisher."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.InteractiveNav.evaluation.restricted_gt_perception import (
    ForbiddenField,
    OpaqueEpisodeRegistry,
    PrivateObjectSpec,
    audit_restricted_gt_payload,
    binary_mask_rle_stats,
    build_restricted_gt_frame,
    decode_binary_mask_rle,
    encode_binary_mask_rle,
)


def test_mask_rle_round_trip_uses_coco_column_order() -> None:
    mask = np.asarray(
        [[False, True, False], [True, True, False], [False, False, True]], dtype=bool
    )

    encoded = encode_binary_mask_rle(mask)

    assert encoded["size"] == [3, 3]
    assert np.array_equal(decode_binary_mask_rle(encoded), mask)
    assert binary_mask_rle_stats(encoded) == (3, 3, int(mask.sum()))


def test_public_frame_is_opaque_and_allow_list_only() -> None:
    segmentation = np.zeros((4, 5, 2), dtype=np.int32)
    segmentation[..., 1] = -1
    segmentation[1:3, 2:5, 0] = 9
    segmentation[1:3, 2:5, 1] = 42
    private_name = "Fridge_opaque_source_007"
    registry = OpaqueEpisodeRegistry()

    payload = build_restricted_gt_frame(
        segmentation=segmentation,
        registry=registry,
        candidates=[
            PrivateObjectSpec(
                source_name=private_name,
                semantic_category="Fridge",
                geom_ids=(9,),
                aabb_center=(1.0, 2.0, 3.0),
                aabb_size=(4.0, 5.0, 6.0),
            )
        ],
        geom_object_type=42,
    )

    audit_restricted_gt_payload(payload, known_private_identifiers=[private_name])
    assert set(payload) == {
        "protocol_version",
        "episode_id",
        "episode_reset",
        "frame_index",
        "observations",
    }
    observation = payload["observations"][0]
    assert set(observation) == {"instance_id", "name", "bbox_2d_xyxy", "mask_rle", "bbox_3d"}
    assert observation["instance_id"] == "obj_000001"
    assert observation["name"] == "refrigerator"
    assert private_name not in repr(payload)
    assert observation["bbox_2d_xyxy"] == [2, 1, 4, 2]
    assert registry.resolve_private_source_name("obj_000001") == private_name


def test_audit_rejects_private_joint_field() -> None:
    registry = OpaqueEpisodeRegistry()
    payload = {
        "protocol_version": "interactive_nav_v3_restricted_gt_v1",
        "episode_id": registry.episode_id,
        "episode_reset": True,
        "frame_index": 0,
        "observations": [
            {
                "instance_id": "obj_000001",
                "name": "door",
                "bbox_2d_xyxy": [0, 0, 0, 0],
                "mask_rle": {"size": [1, 1], "counts": [0, 1]},
                "bbox_3d": {"center": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0], "frame_id": "world"},
                "joint_value": 0.0,
            }
        ],
    }

    with pytest.raises(ForbiddenField, match="joint_value"):
        audit_restricted_gt_payload(payload)


def test_registry_reset_invalidates_previous_episode_ids() -> None:
    registry = OpaqueEpisodeRegistry()
    old_id = registry.public_id_for("private_a")
    old_episode = registry.episode_id

    new_episode = registry.reset()

    assert new_episode != old_episode
    assert registry.resolve_private_source_name(old_id) is None
    assert registry.public_id_for("private_a") == "obj_000001"


def test_restricted_gt_filters_by_camera_distance_and_projected_bbox_area() -> None:
    segmentation = np.zeros((8, 8, 2), dtype=np.int32)
    segmentation[..., 1] = -1
    segmentation[0:3, 0:3, 0] = 1
    segmentation[0:3, 0:3, 1] = 42
    segmentation[0:3, 4:7, 0] = 2
    segmentation[0:3, 4:7, 1] = 42
    segmentation[6:7, 0:1, 0] = 3
    segmentation[6:7, 0:1, 1] = 42

    payload = build_restricted_gt_frame(
        segmentation=segmentation,
        registry=OpaqueEpisodeRegistry(),
        candidates=[
            PrivateObjectSpec(
                source_name="near_large",
                semantic_category="cabinet",
                geom_ids=(1,),
                aabb_center=(2.0, 0.0, 0.0),
            ),
            PrivateObjectSpec(
                source_name="far_large",
                semantic_category="refrigerator",
                geom_ids=(2,),
                aabb_center=(6.75, 0.0, 0.0),
            ),
            PrivateObjectSpec(
                source_name="near_tiny",
                semantic_category="cup",
                geom_ids=(3,),
                aabb_center=(1.0, 0.0, 0.0),
            ),
        ],
        geom_object_type=42,
        min_visible_pixels=1,
        min_bbox_area_pixels=4,
        max_distance_m=4.0,
        camera_position=(0.0, 0.0, 0.0),
    )

    assert [item["name"] for item in payload["observations"]] == ["cabinet"]

    unbounded_payload = build_restricted_gt_frame(
        segmentation=segmentation,
        registry=OpaqueEpisodeRegistry(),
        candidates=[
            PrivateObjectSpec(
                source_name="far_large",
                semantic_category="refrigerator",
                geom_ids=(2,),
                aabb_center=(6.75, 0.0, 0.0),
            )
        ],
        geom_object_type=42,
        min_visible_pixels=1,
        min_bbox_area_pixels=4,
        max_distance_m=0.0,
    )

    assert [item["name"] for item in unbounded_payload["observations"]] == [
        "refrigerator"
    ]
