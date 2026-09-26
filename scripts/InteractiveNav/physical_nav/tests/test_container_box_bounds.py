import numpy as np
import pytest

from physical_yoloe_bridge import _is_implausibly_large_object


@pytest.mark.parametrize("label", ["cabinet", "drawer", "locker", "wardrobe", "closet", "cupboard", "dresser"])
def test_storage_furniture_does_not_bypass_wall_sized_geometry_rejection(label):
    config = {"max_container_box_span_m": 3.5, "max_container_box_volume_m3": 8.0}
    assert _is_implausibly_large_object(label, np.array([5.3, 0.8, 2.0]), config)
    assert not _is_implausibly_large_object(label, np.array([0.8, 0.6, 1.8]), config)
    assert _is_implausibly_large_object(label, np.array([2.5, 2.0, 2.0]), config)


@pytest.mark.parametrize("label", ["door", "portal", "fridge", "refrigerator"])
def test_container_size_gate_does_not_change_door_or_fridge_admission(label):
    assert not _is_implausibly_large_object(label, np.array([6.0, 2.0, 2.0]), {})
