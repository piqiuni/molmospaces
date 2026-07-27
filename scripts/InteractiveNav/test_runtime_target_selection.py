import json

import pytest

from runtime_target_selection import _is_inside, load_fixed_container_target


def test_load_fixed_container_target(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "target_name": "apple_instance",
                "container_name": "fridge_instance",
                "target_context": {
                    "enabled": True,
                    "target_name": "apple",
                    "object_labels": ["apple"],
                },
            }
        ),
        encoding="utf-8",
    )

    context, selection = load_fixed_container_target(path)

    assert context["target_name"] == "apple"
    assert selection["target_name"] == "apple_instance"
    assert selection["selection_mode"] == "fixed_container_object"
    assert selection["selection_input_path"] == str(path.resolve())


def test_load_fixed_container_target_rejects_disabled_context(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"target_context": {"enabled": False}}), encoding="utf-8")

    with pytest.raises(ValueError, match="not enabled"):
        load_fixed_container_target(path)


def test_strict_containment_rejects_object_on_container_top():
    inside, overlap_ratio = _is_inside(
        container_center=[0.0, 0.0, 0.5],
        container_size=[1.0, 1.0, 1.0],
        object_center=[0.0, 0.0, 1.01],
        object_size=[0.2, 0.2, 0.04],
    )

    assert inside is False
    assert overlap_ratio < 0.5


def test_strict_containment_accepts_object_inside_container_volume():
    inside, overlap_ratio = _is_inside(
        container_center=[0.0, 0.0, 0.5],
        container_size=[1.0, 1.0, 1.0],
        object_center=[0.1, 0.0, 0.55],
        object_size=[0.2, 0.2, 0.2],
    )

    assert inside is True
    assert overlap_ratio == pytest.approx(1.0)
