"""Wire-compatible occupancy messages without per-cell Python serialization."""

import numpy as np
from nav_msgs.msg import OccupancyGrid
from rospy.numpy_msg import numpy_msg


NumpyOccupancyGrid = numpy_msg(OccupancyGrid)


def occupancy_data_snapshot(data):
    values = np.asarray(data)
    if values.ndim != 1:
        raise ValueError("occupancy data must be one-dimensional")
    if values.dtype.kind not in "biu":
        values = np.asarray([int(value) for value in data])
    # Validate before the int8 cast, including unsigned and Python-sized ints.
    if values.size and (values.min() < -128 or values.max() > 127):
        raise ValueError("occupancy data is outside the signed int8 wire range")
    result = np.array(values, dtype=np.int8, copy=True)
    result.flags.writeable = False
    return result
