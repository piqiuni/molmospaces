from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest

rospy = pytest.importorskip("rospy")
from nav_msgs.msg import OccupancyGrid
from rospy.msg import args_kwds_to_message
from semantic_mapping_py_pkg.occupancy_transport import (
    NumpyOccupancyGrid,
    occupancy_data_snapshot,
)
from semantic_mapping_py_pkg.semantic_occ_overlay import SemanticOccupancyOverlay


def serialized(message):
    stream = BytesIO()
    message.serialize(stream)
    return stream.getvalue()


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, None])
@pytest.mark.parametrize("seq", [0, 104])
def test_wire_bytes_and_both_receivers_match(dtype, seq):
    data = [-128, -1, 0, 1, 50, 99, 100, 127]
    raw = OccupancyGrid()
    raw.header.seq = seq
    raw.header.frame_id = "map"
    raw.header.stamp = rospy.Time(123, 456)
    raw.info.width, raw.info.height = 4, 2
    raw.info.resolution = 0.05
    raw.info.origin.position.x = -12.5
    raw.info.origin.orientation.w = 1.0
    raw.data = data
    source = np.array(data, dtype=dtype) if dtype else data
    fast = NumpyOccupancyGrid(header=raw.header, info=raw.info,
                              data=occupancy_data_snapshot(source))
    assert isinstance(fast, OccupancyGrid)
    assert args_kwds_to_message(OccupancyGrid, (fast,), {}) is fast
    assert serialized(fast) == serialized(raw)
    expected = OccupancyGrid().deserialize(serialized(raw))
    for receiver in (OccupancyGrid(), NumpyOccupancyGrid()):
        receiver.deserialize(serialized(fast))
        assert list(receiver.data) == data
        assert receiver.header == raw.header
        assert receiver.info == expected.info
    assert not fast.data.flags.writeable
    source[0] = 42
    assert fast.data[0] == -128


@pytest.mark.parametrize("data", [[], [True, False], ["-1", "100"], [1.9, -1.2],
                                  np.array([0, 127], dtype=np.uint64)])
def test_snapshot_retains_int_conversion(data):
    snapshot = occupancy_data_snapshot(data)
    assert snapshot.dtype == np.int8
    assert snapshot.tolist() == list(map(int, data))
    assert snapshot.flags.owndata and not snapshot.flags.writeable


@pytest.mark.parametrize("data", [[-129], [128], [2**80], [float("nan")],
                                  [float("inf")], [[1]], ["invalid"],
                                  np.array([2**64 - 1], dtype=np.uint64)])
def test_invalid_wire_values_are_not_silently_wrapped(data):
    with pytest.raises((ValueError, OverflowError)):
        occupancy_data_snapshot(data)


def test_mapping_copy_preserves_source_metadata():
    from semantic_mapping_node import SemanticMappingNode

    node = object.__new__(SemanticMappingNode)
    node.world_frame = "fallback"
    raw = OccupancyGrid()
    raw.header.seq, raw.header.stamp = 52, rospy.Time(32, 5)
    raw.info.width, raw.info.height = 2, 2
    source = [-1, 0, 99, 100]
    copy = node._build_occupancy_copy(source, raw=raw)
    assert isinstance(copy, NumpyOccupancyGrid)
    assert copy.header.seq == 52 and copy.header.stamp == raw.header.stamp
    assert copy.header.frame_id == "fallback" and copy.info == raw.info
    source[0] = 0
    assert copy.data.tolist() == [-1, 0, 99, 100]


def test_array_overlay_preserves_state_and_python_integer_output():
    info = SimpleNamespace(width=20, height=20, resolution=0.1,
                           origin=SimpleNamespace(position=SimpleNamespace(x=0, y=0),
                                                  orientation=SimpleNamespace(x=0, y=0, z=0, w=1)))
    graph = {"nodes": [{"id": "door", "type": "portal", "aabb_center": [1, 1, 1],
                         "aabb_size": [0.8, 0.1, 2], "interaction": {"state": "open"}}]}
    plain, array = SemanticOccupancyOverlay(), SemanticOccupancyOverlay()
    plain.update_graph(graph)
    array.update_graph(graph)
    for value in (100, 0, 0, 0, 100):
        data = [value] * 400
        actual = array.apply(info, occupancy_data_snapshot(data))
        assert actual == plain.apply(info, data)
        assert all(type(v) is int for v in actual[0])
        assert array.raw_free_streaks == plain.raw_free_streaks
