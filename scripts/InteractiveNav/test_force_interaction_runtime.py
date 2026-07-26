from force_interaction_runtime import (
    build_articulation_targets,
    joint_open_fraction,
)


def _joints():
    return [
        {
            "joint_name": "outer_hinge",
            "joint_type": "hinge",
            "joint_range": [0.0, 1.57],
        },
        {
            "joint_name": "drawer_top",
            "joint_type": "slide",
            "joint_range": [0.0, 0.45],
        },
        {
            "joint_name": "drawer_bottom",
            "joint_type": "slide",
            "joint_range": [0.0, 0.45],
        },
    ]


def test_fridge_style_command_opens_all_joints():
    targets, selected, closed = build_articulation_targets(_joints())

    assert selected == ["outer_hinge", "drawer_top", "drawer_bottom"]
    assert closed == []
    assert targets == {
        "outer_hinge": 1.57,
        "drawer_top": 0.45,
        "drawer_bottom": 0.45,
    }


def test_drawer_style_command_opens_one_slide_and_closes_other_slides():
    targets, selected, closed = build_articulation_targets(
        _joints(),
        selected_joint_names=["outer_hinge", "drawer_top"],
        close_other_joints=True,
    )

    assert selected == ["outer_hinge", "drawer_top"]
    assert closed == ["drawer_bottom"]
    assert targets["outer_hinge"] == 1.57
    assert targets["drawer_top"] == 0.45
    assert targets["drawer_bottom"] == 0.0


def test_open_fraction_uses_closed_endpoint_for_slide():
    assert joint_open_fraction(0.0, [0.0, 0.45]) == 0.0
    assert joint_open_fraction(0.45, [0.0, 0.45]) == 1.0
