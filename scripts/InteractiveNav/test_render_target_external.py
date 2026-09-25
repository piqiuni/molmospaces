import pytest

from scripts.InteractiveNav.render_target_external import camera_settings, structural_hidden


def test_camera_faces_container_from_frozen_approach():
    episode = {"interactive_nav": {"target": {"container_aabb_center": [2, 3, 1]}, "oracle_plan": {"steps": [{"reason": "approach_container_interaction", "goal_point": [2, 5, 0]}]}}}
    settings = camera_settings(episode, 4, -35, None)
    assert settings["azimuth"] == pytest.approx(270)
    assert settings["lookat"] == [2, 3, 1]
    assert settings["direction_source"] == "frozen_container_approach_direction"


def test_explicit_camera_and_target_without_container():
    episode = {"interactive_nav": {"target": {"object_aabb_center": [0, 1, 2]}}}
    assert camera_settings(episode, 3, -40, -90)["azimuth"] == 270
    assert camera_settings(episode, 3, -40, None)["direction_source"] == "fallback_positive_y"


@pytest.mark.parametrize("name,hide_walls,expected", [("ceiling_2_visual", False, True), ("roof_1", False, True), ("wall_4_2_visual", False, False), ("wall_4_2_visual", True, True), ("room_2_visual", True, False), ("cabinet_roof_mesh", True, False)])
def test_structural_filter_preserves_furniture_and_floor(name, hide_walls, expected):
    assert structural_hidden(name, hide_walls) == expected
