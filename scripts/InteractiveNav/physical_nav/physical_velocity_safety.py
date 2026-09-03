#!/usr/bin/env python3
"""ROS node applying the physical Go2 velocity safety profile."""

from __future__ import annotations

import rospy
from geometry_msgs.msg import Twist

from velocity_safety import VelocitySafetyConfig, VelocitySafetyLimiter


class PhysicalVelocitySafetyNode:
    def __init__(self) -> None:
        section = rospy.get_param("~velocity_safety", {})
        if not isinstance(section, dict):
            section = {}

        def setting(name: str, default):
            return rospy.get_param(f"~{name}", section.get(name, default))

        input_topic = str(setting("input_topic", "/physical_nav/shadow_cmd_vel"))
        output_topic = str(setting("output_topic", "/physical_nav/actuated_cmd_vel"))
        config = VelocitySafetyConfig(
            scale=float(setting("scale", 1.0)),
            max_linear_mps=float(setting("max_linear_mps", 0.50)),
            min_linear_mps=float(setting("min_linear_mps", 0.30)),
            linear_deadband_mps=float(setting("linear_deadband_mps", 0.05)),
            max_angular_rps=float(setting("max_angular_rps", 1.20)),
            min_angular_rps=float(setting("min_angular_rps", 0.20)),
            angular_deadband_rps=float(setting("angular_deadband_rps", 0.05)),
            max_linear_accel_mps2=float(setting("max_linear_accel_mps2", 0.15)),
            max_angular_accel_rps2=float(setting("max_angular_accel_rps2", 2.00)),
            stale_after_s=float(setting("stale_after_s", 0.50)),
        )
        self.limiter = VelocitySafetyLimiter(config)
        self.publisher = rospy.Publisher(output_topic, Twist, queue_size=1)
        self.last_input_at = rospy.Time.now()
        self.watchdog = rospy.Timer(rospy.Duration(0.10), self._watchdog)
        rospy.Subscriber(input_topic, Twist, self._callback, queue_size=1)
        rospy.loginfo(
            "physical velocity safety: %s -> %s, scale=%.2f caps=(%.3f, %.3f)",
            input_topic, output_topic, config.scale,
            config.max_linear_mps, config.max_angular_rps,
        )

    def _callback(self, message: Twist) -> None:
        self.last_input_at = rospy.Time.now()
        vx, vy, wz = self.limiter.limit(
            message.linear.x, message.linear.y, message.angular.z
        )
        output = Twist()
        output.linear.x, output.linear.y, output.angular.z = vx, vy, wz
        self.publisher.publish(output)

    def _watchdog(self, _event) -> None:
        age = (rospy.Time.now() - self.last_input_at).to_sec()
        if age <= self.limiter.config.stale_after_s:
            return
        self.publisher.publish(Twist())
        self.limiter.stop()


if __name__ == "__main__":
    rospy.init_node("physical_velocity_safety")
    PhysicalVelocitySafetyNode()
    rospy.spin()
