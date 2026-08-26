from force_interaction_runtime import (
    build_articulation_targets,
    ground_drawer_open_regions,
    joint_open_fraction,
    merge_door_leaf_joint_records,
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


def test_drawer_scan_grounding_uses_visual_top_to_bottom_order():
    groups = ground_drawer_open_regions(
        [
            {"joint_name": "drawer_top", "joint_type": "slide"},
            {"joint_name": "drawer_bottom", "joint_type": "slide"},
        ],
        [
            {"center": [0.5, 0.82]},
            {"center": [0.5, 0.18]},
        ],
        {"drawer_top": 1.2, "drawer_bottom": 0.5},
        fallback_to_all=False,
    )

    assert [group["joint_names"] for group in groups] == [
        ["drawer_top"],
        ["drawer_bottom"],
    ]
    assert all(group["grounding_source"] == "visual_region_vertical_order" for group in groups)


def test_mllm_drawer_scan_does_not_invent_hidden_drawers_without_regions():
    groups = ground_drawer_open_regions(
        [{"joint_name": "drawer_top", "joint_type": "slide"}],
        [],
        {"drawer_top": 1.2},
        fallback_to_all=False,
    )

    assert groups == []


def test_double_door_merges_missing_leaf_and_excludes_handle():
    leaves = merge_door_leaf_joint_records(
        [
            {
                "leaf_body_name": "door_leaf_left",
                "hinge_joint_index": 0,
                "hinge_joint_name": "door_left_joint",
                "joint_range": [0.0, 1.57],
                "joint_id": 1,
            }
        ],
        [
            {
                "joint_name": "door_left_joint",
                "joint_type": "hinge",
                "joint_range": [0.0, 1.57],
                "joint_id": 1,
                "body_name": "door_leaf_left",
            },
            {
                "joint_name": "door_right_joint",
                "joint_type": "hinge",
                "joint_range": [0.0, 1.57],
                "joint_id": 2,
                "body_name": "door_leaf_right",
            },
            {
                "joint_name": "door_handle_joint",
                "joint_type": "hinge",
                "joint_range": [0.0, 1.57],
                "joint_id": 3,
                "body_name": "door_handle",
            },
        ],
    )

    assert [leaf["hinge_joint_name"] for leaf in leaves] == [
        "door_left_joint",
        "door_right_joint",
    ]
