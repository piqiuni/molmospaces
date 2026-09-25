"""Wire-compatible occupancy messages without per-cell Python serialization."""

import numpy as np
import copy
from functools import lru_cache
from nav_msgs.msg import OccupancyGrid
from rospy.numpy_msg import numpy_msg


class _LegacyWireArray(np.ndarray):
    # ROS Noetic generated serializers still use the removed NumPy alias.
    def tostring(self, order="C"):
        return self.tobytes(order=order)


@lru_cache(maxsize=16)
def numpy_msg_compat(message_type):
    base = numpy_msg(message_type)

    class CompatibleNumpyMessage(base):
        # genpy constructors/copy inspect these slots, not the base's slots.
        __slots__ = message_type.__slots__

        def serialize(self, stream):
            values = self.data
            if isinstance(values, np.ndarray) and not hasattr(values, "tostring"):
                # Do not mutate a shared/latched message during serialization.
                proxy = copy.copy(self)
                proxy.data = values.view(_LegacyWireArray)
                return base.serialize(proxy, stream)
            return base.serialize(self, stream)

    return CompatibleNumpyMessage


NumpyOccupancyGrid = numpy_msg_compat(OccupancyGrid)


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
