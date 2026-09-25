from types import SimpleNamespace

from scripts.InteractiveNav.evaluation.scene_distractor_filter import (
    apply_same_category_distractor_filter,
)


def make_spec():
    return SimpleNamespace(
        task={"pickup_obj_name": "apple_target"},
        task_relevant_objects=["apple_task"],
        language=None,
        scene_modifications=SimpleNamespace(
            object_poses={name: [0, 0, 0] for name in (
                "apple_target", "apple_other", "apple_task", "apple_plan", "apple_added", "chair_1",
            )},
            added_objects={"apple_added": {}},
            removed_objects=[],
        ),
    )


def test_filter_protects_target_task_plan_and_added_objects():
    spec = make_spec()
    nav = {
        "target": {"selected_instance": "apple_target", "category": "apple"},
        "interactions": [{"object_name": "apple_plan"}],
    }
    report = apply_same_category_distractor_filter(spec, nav)
    assert report["removed_object_names"] == ["apple_other"]
    assert spec.scene_modifications.removed_objects == ["apple_other"]
    assert set(spec.scene_modifications.object_poses) == {
        "apple_target", "apple_task", "apple_plan", "apple_added", "chair_1",
    }
    assert nav["target"]["selected_instance"] == "apple_target"
    assert apply_same_category_distractor_filter(spec, nav)["removed_object_count"] == 0


def test_filter_disabled_does_not_modify_scene():
    spec = make_spec()
    report = apply_same_category_distractor_filter(
        spec, {"target": {"category": "apple"}}, enabled=False,
    )
    assert report["reason"] == "disabled"
    assert "apple_other" in spec.scene_modifications.object_poses
    assert spec.scene_modifications.removed_objects == []


def test_filter_includes_static_names_aliases_and_namespaced_target():
    spec = make_spec()
    nav = {"target": {"selected_instance": "compactdisk_target", "category": "cd"}}
    report = apply_same_category_distractor_filter(
        spec, nav,
        candidate_names={"scene/compactdisk_target", "scene/compactdisk_other", "chair_1"},
    )
    assert report["removed_object_names"] == ["scene/compactdisk_other"]
