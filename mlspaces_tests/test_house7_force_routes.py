from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "InteractiveNav"
for path in (REPO_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.InteractiveNav.sample_house7_force_routes import (
    build_route_config,
    candidate_approach_points,
    far_goal_candidates,
    load_reused_results,
    parse_seed_spec,
    portal_room_sides,
)


class FakeSceneMap:
    def __init__(self) -> None:
        self.px_per_m = 10.0
        self.occupancy = np.ones((80, 100), dtype=bool)
        self.occupancy[:, 49:51] = False
        self.occupancy[35:45, 49:51] = True
        self.room_map = np.zeros_like(self.occupancy, dtype=int)
        self.room_map[:, :49] = 11
        self.room_map[:, 51:] = 22
        self.room_map[35:45, 49:51] = 22
        self.room_ids_to_name = {11: "left_room", 22: "right_room"}

    def pos_m_to_px(self, point):
        point = np.asarray(point)
        return np.array([round(point[1] * 10), round(point[0] * 10)], dtype=int)

    def pos_px_to_m(self, pixel):
        pixel = np.asarray(pixel)
        if pixel.ndim == 1:
            return np.array([pixel[1] / 10, pixel[0] / 10, 0.0])
        return np.column_stack(
            (pixel[:, 1] / 10, pixel[:, 0] / 10, np.zeros(len(pixel)))
        )


def test_parse_seed_ranges_and_deduplicates() -> None:
    assert parse_seed_spec("0:3,2,5:9:2") == [0, 1, 2, 5, 7]


def test_portal_sides_and_far_goal_stay_in_opposite_room() -> None:
    scene_map = FakeSceneMap()
    sides = portal_room_sides(
        scene_map,
        portal_center_xy=[5.0, 4.0],
        portal_normal_xy=[1.0, 0.0],
        portal_half_width_m=0.5,
    )
    assert sides[-1]["room_id"] == 11
    assert sides[1]["room_id"] == 22

    candidates = far_goal_candidates(
        scene_map,
        room_id=22,
        portal_center_xy=[5.0, 4.0],
        connected_from_xy=[5.5, 4.0],
        min_portal_distance_m=2.0,
        min_clearance_m=0.4,
    )
    assert candidates
    assert all(candidate["xy"][0] > 7.0 for candidate in candidates)


def test_approach_candidates_respect_door_standoff() -> None:
    candidates = candidate_approach_points(
        portal_center_xy=[5.0, 4.0],
        portal_normal_xy=[1.0, 0.0],
        portal_half_width_m=1.0,
        side_sign=-1,
        min_standoff_m=1.15,
    )
    distances = [float(np.linalg.norm(candidate - np.array([5.0, 4.0]))) for candidate in candidates]
    assert min(distances) >= 1.15


def test_route_config_freezes_force_backend_and_atomic_feedback_contract() -> None:
    spec = {
        "scene_dataset": "procthor-10k",
        "data_split": "train",
        "house_ind": 7,
        "variant": "base",
        "robot": "rby1",
        "target_root": "double_door",
        "px_per_m": 80,
        "downscale": 4,
        "min_first_leg_m": 2.0,
        "min_door_standoff_m": 1.15,
        "min_total_route_m": 5.0,
        "min_far_goal_distance_m": 2.5,
        "min_far_goal_clearance_m": 0.45,
    }
    route = {
        "seed": 3,
        "pickup_obj_name": "bed",
        "robot_base_pose": [1, 2, 0.1, 1, 0, 0, 0],
        "start_xyyaw": [1, 2, 0],
        "door_approach_xyyaw": [4, 5, 0],
        "portal_center_xy": [5, 5],
        "portal_normal_xy": [-1, 0],
        "portal_half_width_m": 1.0,
        "far_goal_xyyaw": [8, 5, 0],
        "start_room_id": 1,
        "goal_room_id": 2,
        "start_room_name": "room_a",
        "goal_room_name": "room_b",
        "start_side_sign": -1,
        "first_leg_path_length_m": 3.0,
        "door_standoff_m": 1.15,
        "third_leg_path_length_m": 4.0,
        "total_route_path_length_m": 7.0,
        "far_goal_portal_distance_m": 3.5,
        "far_goal_clearance_m": 0.8,
        "closed_start_to_far_path_found": False,
        "open_approach_to_far_path_found": True,
        "start_collision": False,
    }
    config = build_route_config(spec, [route])
    interaction = config["routes"][0]["interaction"]
    assert interaction["backend"] == "force"
    assert interaction["atomic_sim_steps"] == 1
    assert interaction["assume_success"] is True
    assert config["sampling"]["min_door_standoff_m"] == 1.15


def test_reused_diagnostics_require_matching_sampling_spec(tmp_path) -> None:
    spec = {"house_ind": 7, "min_total_route_m": 5.0}
    path = tmp_path / "diagnostics.json"
    path.write_text(
        json.dumps(
            {
                "spec": spec,
                "results": [
                    {"seed": 1, "valid": True},
                    {"seed": 2, "valid": False},
                ],
            }
        )
    )
    reused = load_reused_results([path], spec, {2, 3})
    assert set(reused) == {2}


def test_frozen_house7_routes_keep_safe_door_standoff() -> None:
    config_path = (
        REPO_ROOT
        / "scripts"
        / "InteractiveNav"
        / "configs"
        / "semantic_decision"
        / "house7_force_routes.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    minimum = float(payload["sampling"]["min_door_standoff_m"])
    assert len(payload["routes"]) == 6
    assert all(
        float(route["validation"]["door_standoff_m"]) + 1e-6 >= minimum
        for route in payload["routes"]
    )
