#!/usr/bin/env python3
"""Read-only ROS OCC throughput/latency probe; count unique source stamps."""
import argparse
import json
import threading
import time

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    rospy.init_node("measure_occ_pipeline", anonymous=True, disable_signals=True)
    samples = {}
    lock = threading.Lock()
    topics = {
        "/physical_nav/points": PointCloud2,
        "/physical_nav/filtered_pointcloud": PointCloud2,
        "/physical_nav/occupancy": OccupancyGrid,
        "/physical_nav/planning_occupancy": OccupancyGrid,
        "/physical_nav/room_segment_grid": OccupancyGrid,
    }

    def callback(msg, topic):
        with lock:
            samples.setdefault(topic, []).append((time.monotonic(), msg.header.stamp.to_nsec(),
                                                  (rospy.Time.now() - msg.header.stamp).to_sec() * 1000))

    subscribers = [rospy.Subscriber(topic, kind, callback, callback_args=topic,
                                   queue_size=1, buff_size=8 * 1024 * 1024,
                                   tcp_nodelay=True) for topic, kind in topics.items()]
    deadline = time.monotonic() + args.seconds
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        time.sleep(0.1)
    for subscriber in subscribers:
        subscriber.unregister()
    with lock:
        result = {}
        for topic, rows in samples.items():
            unique = list({row[1]: row for row in reversed(rows)}.values())
            unique.sort()
            elapsed = rows[-1][0] - rows[0][0]
            ages = [row[2] for row in unique]
            intervals = np.diff([row[0] for row in unique]) * 1000
            result[topic] = dict(messages=len(rows), unique_frames=len(unique),
                                 hz=(len(rows)-1)/max(elapsed, 1e-6),
                                 unique_hz=(len(unique)-1)/max(elapsed, 1e-6),
                                 age_ms_p50_p95=np.percentile(ages, [50, 95]).tolist(),
                                 interval_ms_p50_p95=np.percentile(intervals, [50, 95]).tolist()
                                 if len(intervals) else [])
        upstream = {row[1]: row[0] for row in samples.get('/physical_nav/points', [])}
        delays = [(row[0] - upstream[row[1]]) * 1000
                  for row in samples.get('/physical_nav/occupancy', []) if row[1] in upstream]
        if delays:
            result['points_to_occ_ms_p50_p95'] = np.percentile(delays, [50, 95]).tolist()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
