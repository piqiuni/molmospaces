"""Native rospy counterpart for :mod:`roslibpy_ros_smoke`."""

from __future__ import annotations

import rospy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage


class SmokeNode:
    def __init__(self) -> None:
        self.odom_received = False
        self.tf_received = False
        self.cloud_received = False
        self.command_sent = False
        self.command_pub = rospy.Publisher("/cmd_vel_stamped", TwistStamped, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self._odom, queue_size=1)
        rospy.Subscriber("/tf", TFMessage, self._tf, queue_size=1)
        rospy.Subscriber("/registered_scan", PointCloud2, self._cloud, queue_size=1)

    def _odom(self, msg: Odometry) -> None:
        if msg.header.frame_id == "tf_frame_odom" and msg.child_frame_id == "tf_frame_base_link":
            self.odom_received = True
        self._respond_if_ready()

    def _tf(self, msg: TFMessage) -> None:
        self.tf_received = bool(msg.transforms)
        self._respond_if_ready()

    def _cloud(self, msg: PointCloud2) -> None:
        self.cloud_received = msg.width == 3 and msg.point_step == 12 and len(msg.data) == 36
        self._respond_if_ready()

    def _respond_if_ready(self) -> None:
        if self.command_sent or not (self.odom_received and self.tf_received and self.cloud_received):
            return
        message = TwistStamped()
        message.header.stamp = rospy.Time.now()
        message.twist.linear.x = 0.25
        message.twist.angular.z = -0.5
        self.command_pub.publish(message)
        self.command_sent = True
        rospy.loginfo("ROS_NATIVE_SMOKE_OK odom=tf cloud=3 cmd_vel=published")


def main() -> None:
    rospy.init_node("molmospaces_native_ros_smoke")
    node = SmokeNode()
    deadline = rospy.Time.now() + rospy.Duration(30.0)
    rate = rospy.Rate(20)
    while not rospy.is_shutdown() and rospy.Time.now() < deadline and not node.command_sent:
        rate.sleep()
    if not node.command_sent:
        raise RuntimeError("Timed out waiting for native messages from roslibpy")


if __name__ == "__main__":
    main()
