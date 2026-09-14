import pytest

from semantic_mapping_py_pkg.messages import observation_stamp_seconds


@pytest.mark.parametrize("payload,expected", [
    ({"stamp_sec": 100, "stamp_nsec": 521682000}, 100.521682),
    ({"stamp_sec": 100, "capture_stamp_sec": 100.521682}, 100.521682),
    ({"stamp_sec": 100.521682}, 100.521682),
    ({"stamp_sec": 0, "stamp_nsec": 0}, 0),
    ({"stamp_sec": float("nan")}, 10),
    ({"stamp_sec": "bad"}, 10),
    ({}, 10),
])
def test_observation_stamp_preserves_subsecond_precision(payload, expected):
    assert observation_stamp_seconds(payload, 10) == pytest.approx(expected)
