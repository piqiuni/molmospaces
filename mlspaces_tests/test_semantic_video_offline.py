from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "InteractiveNav"
for path in (REPO_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.InteractiveNav.build_semantic_video_offline import (
    align_exact_sim_records,
    align_nearest_timestamp_recorder_frames,
    gt_draw_spec,
    index_recorder_frames,
    panel_names,
    route_event_at_stamp,
    route_target_at_stamp,
)


def test_nearest_timestamp_alignment_handles_extra_recorder_callbacks() -> None:
    sim_records = [
        {"step_index": 0, "stamp_sec": 10.0},
        {"step_index": 1, "stamp_sec": 15.0},
    ]
    recorder_records = [
        {"source_step_value": 0, "image_stamp_value": 10.0},
        {"source_step_value": 1, "image_stamp_value": 11.0},
        {"source_step_value": 2, "image_stamp_value": 14.9},
    ]
    aligned = align_nearest_timestamp_recorder_frames(sim_records, recorder_records)
    assert aligned[0]["image_stamp_value"] == 10.0
    assert aligned[1]["image_stamp_value"] == 14.9


def test_recorder_frames_are_indexed_by_exact_source_step() -> None:
    records = [
        {"source_step_value": 0, "image_stamp_value": 10.0},
        {"source_step_value": 1, "image_stamp_value": 10.2},
    ]
    indexed = index_recorder_frames(records)
    assert sorted(indexed) == [0, 1]
    assert indexed[1]["image_stamp_value"] == 10.2


def test_duplicate_source_steps_are_rejected() -> None:
    records = [
        {"source_step_value": 4, "image_stamp_value": 1.0},
        {"source_step_value": 4, "image_stamp_value": 1.1},
    ]
    try:
        index_recorder_frames(records)
    except RuntimeError as exc:
        assert "Duplicate recorder source steps" in str(exc)
    else:
        raise AssertionError("duplicate recorder steps should fail")


def test_only_unmatched_shutdown_tail_is_trimmed() -> None:
    sim_records = [{"step_index": step} for step in range(4)]
    aligned, trimmed = align_exact_sim_records(sim_records, {0: {}, 1: {}, 2: {}})
    assert [record["step_index"] for record in aligned] == [0, 1, 2]
    assert trimmed == [3]
    try:
        align_exact_sim_records(sim_records, {0: {}, 2: {}, 3: {}})
    except RuntimeError as exc:
        assert "Missing exact recorder snapshots" in str(exc)
    else:
        raise AssertionError("interior recorder gaps should fail")


def test_route_event_is_causal_and_panel_layout_is_explicit() -> None:
    events = [
        {"event": "route_started", "wall_time": 5.0},
        {"event": "interaction_succeeded", "wall_time": 8.0},
    ]
    assert route_event_at_stamp(events, 7.0)["event"] == "route_started"
    assert route_event_at_stamp(events, 9.0)["event"] == "interaction_succeeded"
    assert panel_names(3) == (
        ("CAMERA", "OCC", "ROOM + INTERACTION"),
        ("GLOBAL + LOCAL", "SEMANTIC XY", "TOPOLOGY"),
    )


def test_minimal_gt_schema_uses_payload_image_size_and_labels_name() -> None:
    payload = {
        "image_size": [1024, 576],
        "observations": [
            {
                "id": "gt_000001",
                "name": "Door",
                "bbox_2d": [256, 144, 767, 431],
            }
        ],
    }
    spec = gt_draw_spec(
        (360, 640, 3),
        payload,
        payload["observations"][0],
        "gt_000001",
    )
    assert spec is not None
    assert spec["start"] == (160, 90)
    assert spec["end"] == (479, 269)
    assert spec["label"] == "INTERACT Door gt_000001"
    assert spec["color"] == (235, 35, 210)
    assert spec["thickness"] == 4

    payload["observations"][0]["id"] = "doorway_odo234694d8669f8c477500ae8"
    compact_spec = gt_draw_spec(
        (360, 640, 3),
        payload,
        payload["observations"][0],
        "doorway_odo234694d8669f8c477500ae8",
    )
    assert compact_spec is not None
    assert compact_spec["label"] == "INTERACT Door"


def test_route_target_is_carried_from_route_start_into_interaction() -> None:
    events = [
        {"event": "route_started", "wall_time": 5.0, "target_root": "gt_000001"},
        {
            "event": "interaction_started",
            "wall_time": 8.0,
            "command": {"object_id": "gt_000001"},
        },
    ]
    assert route_target_at_stamp(events, 7.0) == "gt_000001"
    assert route_target_at_stamp(events, 9.0) == "gt_000001"
