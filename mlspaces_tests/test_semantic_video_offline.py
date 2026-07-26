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
    index_recorder_frames,
    panel_names,
    route_event_at_stamp,
)


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
