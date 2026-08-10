from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "InteractiveNav"
for path in (REPO_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.InteractiveNav.build_semantic_video_offline import (
    align_exact_sim_records,
    align_nearest_timestamp_recorder_frames,
    episode_trajectory_prefix,
    gt_draw_spec,
    index_recorder_frames,
    load_episode_trajectory,
    offline_display_config,
    panel_names,
    receipt_is_causal_at_boundary,
    resolve_episode_trajectory_path,
    route_event_at_stamp,
    route_target_at_stamp,
    select_causal_receipt,
)
import scripts.InteractiveNav.offline_semantic_renderer as offline_renderer
from scripts.InteractiveNav.offline_semantic_renderer import (
    OfflineSixPanelRenderer,
    RawGrid,
    TransformResolver,
    TransformSample,
    active_semantic_selection,
    camera_title,
    candidate_matches_canonical_selection,
    extend_world_bounds_lower,
    terminal_status_summary,
    zoom_world_bounds,
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


def test_offline_display_defaults_and_persisted_overrides() -> None:
    assert offline_display_config({}) == {
        "global_panel_scale": 1.8,
        "room_panel_scale": 1.5,
        "semantic_xy_panel_scale": 1.8,
        "occ_lower_margin_m": 1.0,
        "semantic_xy_label_mode": "interaction_target_only",
        "semantic_xy_overview_inset": False,
    }
    assert offline_display_config(
        {
            "video_global_panel_scale": 1.2,
            "video_room_panel_scale": 1.6,
            "video_semantic_xy_panel_scale": 2.0,
            "video_occ_lower_margin_m": 1.25,
            "video_semantic_xy_label_mode": "all",
            "video_semantic_xy_overview_inset": True,
        }
    ) == {
        "global_panel_scale": 1.2,
        "room_panel_scale": 1.6,
        "semantic_xy_panel_scale": 2.0,
        "occ_lower_margin_m": 1.25,
        "semantic_xy_label_mode": "all",
        # A historical recorder setting cannot re-enable the obstructive inset;
        # the offline CLI flag is the explicit opt-in.
        "semantic_xy_overview_inset": False,
    }


def test_world_coordinate_zoom_scales_bounds_without_post_render_crop() -> None:
    assert zoom_world_bounds((0.0, 0.0, 12.0, 18.0), 1.5) == (2.0, 3.0, 10.0, 15.0)


def test_occ_lower_margin_extends_only_the_lower_world_bound() -> None:
    assert extend_world_bounds_lower((1.0, 2.0, 7.0, 9.0), 1.25) == (
        1.0,
        0.75,
        7.0,
        9.0,
    )
    assert extend_world_bounds_lower((1.0, 2.0, 7.0, 9.0), 0.0) == (
        1.0,
        2.0,
        7.0,
        9.0,
    )


def test_semantic_xy_target_only_keeps_rooms_and_hides_non_target_labels(monkeypatch) -> None:
    import cv2

    drawn_labels: list[str] = []
    original_put_text = cv2.putText

    def capture_put_text(image, text, *args, **kwargs):
        drawn_labels.append(str(text))
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_put_text)
    renderer = OfflineSixPanelRenderer(
        transforms=TransformResolver([], map_frame="map", odom_frame="odom")
    )
    step = {
        "pose": [5.0, 5.0, 0.0],
        "semantic_selection": {"target_id": "door_0001"},
        "observed_instance_ids": ["door_0001", "bed_0002"],
        "unified_graph": {
            "nodes": [
                {
                    "id": "room_1",
                    "type": "room",
                    "centroid": [5.0, 5.0],
                    "aabb_size": [8.0, 8.0, 0.0],
                    "attributes": {"room_attribute": "bedroom"},
                },
                {
                    "id": "door_0001",
                    "type": "portal",
                    "label": "door_0001",
                    "centroid": [4.0, 5.0],
                    "aabb_size": [0.3, 1.0, 0.0],
                    "attributes": {"object_id": "door_0001"},
                },
                {
                    "id": "bed_0002",
                    "type": "object",
                    "label": "bed",
                    "centroid": [6.0, 5.0],
                    "aabb_size": [1.5, 2.0, 0.0],
                    "attributes": {"object_id": "bed_0002"},
                },
            ],
            "edges": [],
        },
    }
    renderer.render_semantic_xy(
        None,
        (480, 270),
        step,
        0,
        (0.0, 0.0, 10.0, 10.0),
        view_scale=1.8,
        label_mode="interaction_target_only",
    )

    assert "bedroom room" in drawn_labels
    assert "INTERACT #1 door_0001" in drawn_labels
    assert "#2 bed" not in drawn_labels


def test_terminal_selection_clears_stale_goal_from_offline_rendering() -> None:
    step = {
        "pose": [1.0, 1.0, 0.0],
        "active_goal": [8.0, 8.0],
        "semantic_selection": {
            "active": False,
            "candidate_id": "interaction:door_0003:open",
            "target_id": "door_0003",
            "goal_xyyaw": [8.0, 8.0, 0.0],
        },
    }
    assert active_semantic_selection(step) == {}
    assert "dist_to_goal=-" in camera_title(step, 12)


def test_offline_selection_uses_executor_effective_fallback_goal() -> None:
    step = {
        "semantic_selection": {
            "active": True,
            "candidate_id": "interaction:door_0003:open",
            "goal_xyyaw": [6.859, 4.467, -1.57],
        },
        "semantic_execution_state": {
            "state": "APPROACH_INTERACTION",
            "candidate_id": "interaction:door_0003:open",
            "effective_goal_xyyaw": [6.859, 4.967, -1.57],
        },
    }
    assert active_semantic_selection(step)["goal_xyyaw"] == [6.859, 4.967, -1.57]


def test_episode_trajectory_csv_is_causal_and_raw_reference_is_resolved(tmp_path: Path) -> None:
    debug_dir = tmp_path / "debug"
    raw_dir = debug_dir / "raw"
    raw_dir.mkdir(parents=True)
    trajectory_path = debug_dir / "trajectory.csv"
    trajectory_path.write_text(
        "step_id,elapsed_sec,stamp,x,y,yaw\n"
        "0,0.0,10.0,1.0,2.0,0.1\n"
        "2,0.2,10.2,2.0,3.0,0.2\n"
        "5,0.5,10.5,5.0,6.0,0.5\n",
        encoding="utf-8",
    )
    trajectory = load_episode_trajectory(trajectory_path)
    assert episode_trajectory_prefix(trajectory, 1) == [(0, 1.0, 2.0, 0.1)]
    assert episode_trajectory_prefix(trajectory, 2) == [
        (0, 1.0, 2.0, 0.1),
        (2, 2.0, 3.0, 0.2),
    ]
    assert resolve_episode_trajectory_path(
        raw_dir,
        debug_dir,
        [{"trajectory_reference": "../trajectory.csv"}],
    ) == trajectory_path


def test_stale_same_id_candidate_never_replaces_canonical_goal(monkeypatch) -> None:
    canonical = {
        "active": True,
        "candidate_id": "frontier:1:9",
        "behavior_type": "EXPLORE",
        "goal_xyyaw": [1.0, 1.0, 0.0],
        "candidate_revision": "new-geometry",
    }
    stale_candidate = {
        "candidate_id": "frontier:1:9",
        "behavior_type": "EXPLORE",
        "goal_xyyaw": [3.0, 2.0, 0.0],
        "candidate_revision": "old-geometry",
    }
    assert not candidate_matches_canonical_selection(canonical, stale_candidate)

    arrows: list[tuple[tuple[int, int], float]] = []

    def capture_arrow(_panel, center, yaw, _length, _color):
        arrows.append((center, yaw))

    monkeypatch.setattr(offline_renderer, "_draw_goal_arrow", capture_arrow)
    grid = RawGrid(
        values=np.zeros((120, 120), dtype=np.int32),
        width=120,
        height=120,
        resolution=0.1,
        frame_id="odom",
        origin_x=-6.0,
        origin_y=-6.0,
        origin_yaw=0.0,
    )
    renderer = OfflineSixPanelRenderer(
        transforms=TransformResolver(
            [TransformSample(step_index=0, x=1.0, y=0.0, yaw=math.pi / 2.0)],
            map_frame="map",
            odom_frame="odom",
        )
    )
    renderer.render_map_panel(
        grid,
        (480, 270),
        {
            "pose": [0.0, 0.0, 0.0],
            "semantic_selection": canonical,
            "semantic_candidates": {"candidates": [stale_candidate]},
        },
        0,
        title="OCC",
        kind="occupancy",
        # One transformed corner lies just outside this grid. The renderer
        # must still clip the world bounds and retain the canonical goal.
        world_bounds=(-5.0, -5.0, 5.0, 5.0),
        draw_semantic_candidates=True,
    )
    # There is one arrow, from the canonical map goal transformed into odom.
    # Its yaw proves that local panels use the transformed, not map-frame yaw.
    assert len(arrows) == 1
    assert math.isclose(arrows[0][1], -math.pi / 2.0, abs_tol=1e-6)


def test_renderer_uses_episode_trajectory_instead_of_boundary_history(monkeypatch) -> None:
    captured: list[list[tuple[int, int]]] = []

    def capture_trail(_panel, points, *_args, **_kwargs):
        captured.append(points)

    monkeypatch.setattr(offline_renderer, "_draw_faded_trajectory", capture_trail)
    grid = RawGrid(
        values=np.zeros((100, 100), dtype=np.int32),
        width=100,
        height=100,
        resolution=0.1,
        frame_id="map",
        origin_x=0.0,
        origin_y=0.0,
        origin_yaw=0.0,
    )
    renderer = OfflineSixPanelRenderer(
        transforms=TransformResolver([], map_frame="map", odom_frame="map")
    )
    renderer.render_map_panel(
        grid,
        (480, 270),
        {
            "pose": [0.2, 0.2, 0.0],
            "trajectory": [(0.0, 0.2, 0.2, 0.0)],
        },
        2,
        title="OCC",
        kind="occupancy",
        episode_trajectory=[
            (0, 0.1, 0.1, 0.0),
            (1, 0.2, 0.2, 0.0),
            (2, 0.3, 0.3, 0.0),
        ],
    )
    assert len(captured) == 1
    assert len(captured[0]) == 3


def test_future_map_receipt_falls_back_to_last_causal_receipt() -> None:
    prior = {
        "receipt_id": "planning_occ:8",
        "stage": "planning_occ",
        "step_index": 8,
        "stamp_sec": 9.95,
        "source_index": 8,
    }
    future = {
        "receipt_id": "planning_occ:9",
        "stage": "planning_occ",
        "step_index": 9,
        "stamp_sec": 10.25,
        "source_index": 9,
    }
    assert receipt_is_causal_at_boundary(
        prior,
        boundary_stamp_sec=10.0,
        boundary_step_index=10,
    )
    assert not receipt_is_causal_at_boundary(
        future,
        boundary_stamp_sec=10.0,
        boundary_step_index=10,
    )
    selected = select_causal_receipt(
        stage="planning_occ",
        requested_receipt="planning_occ:9",
        maps_by_id={prior["receipt_id"]: prior, future["receipt_id"]: future},
        stage_records=[prior, future],
        boundary_stamp_sec=10.0,
        boundary_step_index=10,
    )
    assert selected.requested_receipt_was_future
    assert selected.used_causal_fallback
    assert selected.reason == "requested_future_fallback"
    assert selected.selected_meta == prior


def test_costmap_palette_keeps_soft_inscribed_and_lethal_distinct() -> None:
    grid = RawGrid(
        values=np.asarray([[0, 1, 98, 99, 100]], dtype=np.int32),
        width=5,
        height=1,
        resolution=0.1,
        frame_id="map",
        origin_x=0.0,
        origin_y=0.0,
        origin_yaw=0.0,
    )
    image = offline_renderer._costmap_base(grid)
    assert tuple(image[0, 2]) == offline_renderer.COSTMAP_SOFT_DARK_COLOR
    assert tuple(image[0, 3]) == offline_renderer.COSTMAP_INSCRIBED_COLOR
    assert tuple(image[0, 4]) == offline_renderer.COSTMAP_LETHAL_COLOR
    assert tuple(image[0, 1]) != tuple(image[0, 2])
    assert len(
        {
            tuple(image[0, 2]),
            tuple(image[0, 3]),
            tuple(image[0, 4]),
        }
    ) == 3


def test_unselected_explore_candidates_use_pale_purple(monkeypatch) -> None:
    import cv2

    colors: list[tuple[int, int, int]] = []
    original_circle = cv2.circle

    def capture_circle(image, center, radius, color, *args, **kwargs):
        colors.append(tuple(color))
        return original_circle(image, center, radius, color, *args, **kwargs)

    monkeypatch.setattr(cv2, "circle", capture_circle)
    grid = RawGrid(
        values=np.zeros((100, 100), dtype=np.int32),
        width=100,
        height=100,
        resolution=0.1,
        frame_id="map",
        origin_x=0.0,
        origin_y=0.0,
        origin_yaw=0.0,
    )
    renderer = OfflineSixPanelRenderer(
        transforms=TransformResolver([], map_frame="map", odom_frame="map")
    )
    renderer.render_map_panel(
        grid,
        (480, 270),
        {
            "pose": [5.0, 5.0, 0.0],
            "semantic_selection": {
                "active": True,
                "candidate_id": "frontier:selected",
                "behavior_type": "EXPLORE",
                "goal_xyyaw": [6.0, 6.0, 0.0],
            },
            "semantic_candidates": {
                "candidates": [
                    {
                        "candidate_id": "frontier:selected",
                        "behavior_type": "EXPLORE",
                        "goal_xyyaw": [6.0, 6.0, 0.0],
                    },
                    {
                        "candidate_id": "frontier:other",
                        "behavior_type": "EXPLORE",
                        "goal_xyyaw": [4.0, 4.0, 0.0],
                    },
                ]
            },
        },
        0,
        title="OCC",
        kind="occupancy",
        world_bounds=(0.0, 0.0, 10.0, 10.0),
        draw_semantic_candidates=True,
    )
    assert offline_renderer.UNSELECTED_EXPLORE_COLOR in colors


def test_stale_candidate_text_explains_live_goal_is_authoritative(monkeypatch) -> None:
    import cv2

    labels: list[str] = []
    original_put_text = cv2.putText

    def capture_put_text(image, text, *args, **kwargs):
        labels.append(str(text))
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_put_text)
    grid = RawGrid(
        values=np.zeros((100, 100), dtype=np.int32),
        width=100,
        height=100,
        resolution=0.1,
        frame_id="map",
        origin_x=0.0,
        origin_y=0.0,
        origin_yaw=0.0,
    )
    renderer = OfflineSixPanelRenderer(
        transforms=TransformResolver([], map_frame="map", odom_frame="map")
    )
    renderer.render_map_panel(
        grid,
        (480, 270),
        {
            "pose": [5.0, 5.0, 0.0],
            "semantic_selection": {
                "active": True,
                "candidate_id": "frontier:1:9",
                "behavior_type": "EXPLORE",
                "goal_xyyaw": [6.0, 6.0, 0.0],
                "candidate_revision": "current",
            },
            "semantic_candidates": {
                "candidates": [
                    {
                        "candidate_id": "frontier:1:9",
                        "behavior_type": "EXPLORE",
                        "goal_xyyaw": [4.0, 4.0, 0.0],
                        "candidate_revision": "old",
                    }
                ]
            },
        },
        0,
        title="OCC",
        kind="occupancy",
        world_bounds=(0.0, 0.0, 10.0, 10.0),
        draw_semantic_candidates=True,
    )
    assert "CANDIDATE SNAPSHOT OUTDATED (LIVE GOAL SHOWN)" in labels


def test_interaction_subgoal_is_true_orange_and_legend_uses_live_behavior(monkeypatch) -> None:
    import cv2

    labels: list[str] = []
    original_put_text = cv2.putText

    def capture_put_text(image, text, *args, **kwargs):
        labels.append(str(text))
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_put_text)
    assert offline_renderer.candidate_color("EXPLORE") != offline_renderer.candidate_color(
        "NAVIGATE"
    )
    assert offline_renderer.candidate_color("INTERACT") == (0, 140, 255)
    grid = RawGrid(
        values=np.zeros((100, 100), dtype=np.int32),
        width=100,
        height=100,
        resolution=0.1,
        frame_id="map",
        origin_x=0.0,
        origin_y=0.0,
        origin_yaw=0.0,
    )
    renderer = OfflineSixPanelRenderer(
        transforms=TransformResolver([], map_frame="map", odom_frame="map")
    )
    renderer.render_map_panel(
        grid,
        (480, 270),
        {
            "pose": [5.0, 5.0, 0.0],
            "semantic_selection": {
                "active": True,
                "candidate_id": "interaction:fridge:open",
                "behavior_type": "INTERACT",
                "goal_xyyaw": [6.0, 6.0, 0.0],
            },
            "semantic_candidates": {"candidates": []},
        },
        0,
        title="OCC",
        kind="occupancy",
        world_bounds=(0.0, 0.0, 10.0, 10.0),
        draw_semantic_candidates=True,
    )
    assert "LIVE INTERACT" in labels


def test_terminal_status_uses_recorded_no_plan_countdown_only() -> None:
    summary = terminal_status_summary(
        {
            "semantic_candidates": {
                "exploration_context": {
                    "frontier_exhausted": True,
                    "navigation_frontier_count": 0,
                    "interaction_frontier_count": 0,
                }
            },
            "semantic_decision_trace": {
                "terminal_no_plan_exit": {
                    "armed": True,
                    "complete": False,
                    "detail": {
                        "no_executable_elapsed_steps": 17,
                        "no_executable_candidate_min_steps": 20,
                        "no_executable_observation_confirmations": 2,
                        "no_executable_candidate_confirmations_required": 3,
                    },
                }
            },
        }
    )
    assert summary["remaining_steps"] == 3
    assert summary["remaining_confirmations"] == 1
    assert "STEP 17/20" in str(summary["label"])
    assert "REM 3 STEP / 1 OBS" in str(summary["label"])
    assert terminal_status_summary({}) == {}
