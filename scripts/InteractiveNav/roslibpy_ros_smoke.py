"""End-to-end smoke test for separate MolmoSpaces and RoboStack environments.

Run this program from the ``mlspaces`` environment after the ROS environment
starts ``roscore``, ``rosbridge_websocket`` and the companion ROS smoke node.
It publishes standard Odom, TF and PointCloud2 messages, then verifies that a
native ROS ``TwistStamped`` returns through rosbridge.
"""

from __future__ import annotations

import argparse
import time

from molmo_spaces.policy.learned_policy.roslibpy_bridge import RoslibpyBridgeClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    bridge = RoslibpyBridgeClient(host=args.host, port=args.port)
    try:
        bridge.connect(timeout_s=args.timeout)
        stamp = time.time()
        bridge.publish_odom(
            position_xyz=(1.0, 2.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            stamp_sec=stamp,
        )
        bridge.publish_transform(
            parent_frame="tf_frame_odom",
            child_frame="tf_frame_base_link",
            translation_xyz=(1.0, 2.0, 0.0),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            stamp_sec=stamp,
        )
        bridge.publish_xyz32_pointcloud(
            [(1.0, 0.0, 0.0), (2.0, 0.5, 0.0), (3.0, -0.5, 0.2)],
            stamp_sec=stamp,
        )

        deadline = time.monotonic() + args.timeout
        while bridge.latest_cmd_vel is None and time.monotonic() < deadline:
            time.sleep(0.05)
        command = bridge.latest_cmd_vel
        if command is None:
            raise TimeoutError("No /cmd_vel_stamped was received from ROS")
        if abs(command.linear_x - 0.25) > 1e-6 or abs(command.angular_z + 0.5) > 1e-6:
            raise RuntimeError(f"Unexpected ROS command: {command}")
        print(
            "ROSBRIDGE_SMOKE_OK "
            f"cmd_vel=({command.linear_x:.2f},{command.linear_y:.2f},{command.angular_z:.2f})"
        )
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
