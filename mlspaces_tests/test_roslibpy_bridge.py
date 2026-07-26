from __future__ import annotations

import base64
import struct

from molmo_spaces.policy.learned_policy import roslibpy_bridge


class FakeRos:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.is_connected = False

    def run(self):
        self.is_connected = True

    def terminate(self):
        self.is_connected = False


class FakeTopic:
    def __init__(self, _ros, name: str, message_type: str):
        self.name = name
        self.message_type = message_type
        self.published = []
        self.callback = None

    def publish(self, message):
        self.published.append(message)

    def subscribe(self, callback):
        self.callback = callback

    def unsubscribe(self):
        self.callback = None


class FakeRoslibpy:
    Ros = FakeRos
    Topic = FakeTopic

    @staticmethod
    def Message(value):
        return value


def test_bridge_preserves_native_ros_topic_types(monkeypatch):
    monkeypatch.setattr(roslibpy_bridge, "roslibpy", FakeRoslibpy)
    bridge = roslibpy_bridge.RoslibpyBridgeClient(host="127.0.0.1", port=19090)
    bridge.connect()

    bridge.publish_odom(
        position_xyz=(1.0, 2.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        stamp_sec=100.25,
    )
    bridge.publish_transform(
        parent_frame="tf_frame_odom",
        child_frame="tf_frame_base_link",
        translation_xyz=(1.0, 2.0, 0.0),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        stamp_sec=100.25,
    )
    bridge.publish_xyz32_pointcloud([(1.0, 2.0, 3.0)], stamp_sec=100.25)

    assert bridge._odom_pub.message_type == "nav_msgs/Odometry"
    assert bridge._tf_pub.message_type == "tf2_msgs/TFMessage"
    assert bridge._pointcloud_pub.message_type == "sensor_msgs/PointCloud2"
    assert bridge._odom_pub.published[-1]["header"]["frame_id"] == "tf_frame_odom"
    assert bridge._tf_pub.published[-1]["transforms"][0]["child_frame_id"] == (
        "tf_frame_base_link"
    )

    cloud = bridge._pointcloud_pub.published[-1]
    assert cloud["point_step"] == 12
    assert struct.unpack("<fff", base64.b64decode(cloud["data"])) == (1.0, 2.0, 3.0)

    bridge._cmd_vel_sub.callback(
        {
            "header": {"stamp": {"secs": 5, "nsecs": 500_000_000}},
            "twist": {"linear": {"x": 0.25, "y": 0.0}, "angular": {"z": -0.5}},
        }
    )
    assert bridge.latest_cmd_vel == roslibpy_bridge.TwistCommand(0.25, 0.0, -0.5, 5.5)

    bridge._move_base_status_sub.callback({"status_list": [{"status": 1}]})
    assert bridge.move_base_active is True

    bridge.close()
    assert bridge.is_connected is False
