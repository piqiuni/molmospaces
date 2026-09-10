#!/usr/bin/env python3
"""Select one physical velocity source before the safety limiter.

``move_base`` continuously produces navigation commands while the semantic
executor only occasionally owns the base for an interaction turn/recovery.
Publishing both sources to one ROS topic lets TCPROS interleave messages and
made the downstream safety node apply the wrong command for a cycle.  This
node is the single writer of ``shadow_cmd_vel``.
"""

from __future__ import annotations

import threading

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String


def _is_zero(message: Twist, epsilon: float = 1.0e-4) -> bool:
    return (
        abs(message.linear.x) <= epsilon
        and abs(message.linear.y) <= epsilon
        and abs(message.linear.z) <= epsilon
        and abs(message.angular.x) <= epsilon
        and abs(message.angular.y) <= epsilon
        and abs(message.angular.z) <= epsilon
    )


class VelocityCommandMux:
    """Priority mux with bounded command age and an explicit stop hold."""

    def __init__(self) -> None:
        self.move_base_topic = rospy.get_param(
            "~move_base_topic", "/physical_nav/move_base_cmd_vel"
        )
        self.semantic_topic = rospy.get_param(
            "~semantic_topic", "/physical_nav/semantic_cmd_vel"
        )
        self.output_topic = rospy.get_param(
            "~output_topic", "/physical_nav/shadow_cmd_vel"
        )
        self.stop_topic = rospy.get_param("~stop_topic", "/physical_nav/base_stop")
        self.publish_rate_hz = max(
            5.0, float(rospy.get_param("~publish_rate_hz", 20.0))
        )
        self.move_base_timeout_s = max(
            0.10, float(rospy.get_param("~move_base_timeout_s", 0.35))
        )
        self.semantic_timeout_s = max(
            0.10, float(rospy.get_param("~semantic_timeout_s", 0.40))
        )
        self.semantic_stop_hold_s = max(
            0.05, float(rospy.get_param("~semantic_stop_hold_s", 0.35))
        )

        self._lock = threading.Lock()
        self._move_base = Twist()
        self._semantic = Twist()
        self._move_base_at = rospy.Time(0)
        self._semantic_at = rospy.Time(0)
        self._semantic_active_until = rospy.Time(0)
        self._semantic_stop_until = rospy.Time(0)
        self._explicit_stop_until = rospy.Time(0)
        self._last_source = "none"

        self.publisher = rospy.Publisher(self.output_topic, Twist, queue_size=1)
        rospy.Subscriber(
            self.move_base_topic, Twist, self._move_base_callback, queue_size=1
        )
        rospy.Subscriber(
            self.semantic_topic, Twist, self._semantic_callback, queue_size=1
        )
        rospy.Subscriber(self.stop_topic, String, self._stop_callback, queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate_hz), self._publish
        )
        rospy.loginfo(
            "velocity mux: move_base=%s semantic=%s -> %s at %.1f Hz",
            self.move_base_topic,
            self.semantic_topic,
            self.output_topic,
            self.publish_rate_hz,
        )

    def _stop_callback(self, message: String) -> None:
        if message.data != "stop":
            return
        with self._lock:
            # Explicit terminal stops do not need a prior semantic velocity
            # lease. Drop commands received during the hold so expiry cannot
            # resurrect a residual command from the canceled action.
            self._explicit_stop_until = rospy.Time.now() + rospy.Duration(self.semantic_stop_hold_s)
            self._move_base = Twist()
            self._semantic = Twist()
            self._move_base_at = rospy.Time(0)
            self._semantic_at = rospy.Time(0)
            self._semantic_active_until = rospy.Time(0)

    def _move_base_callback(self, message: Twist) -> None:
        with self._lock:
            if rospy.Time.now() <= self._explicit_stop_until:
                return
            self._move_base = message
            self._move_base_at = rospy.Time.now()

    def _semantic_callback(self, message: Twist) -> None:
        now = rospy.Time.now()
        with self._lock:
            if now <= self._explicit_stop_until:
                return
            was_active = now <= self._semantic_active_until
            self._semantic = message
            self._semantic_at = now
            if _is_zero(message):
                # A stop following a semantic turn must briefly prevent
                # move_base from resuming in the same control cycle.  Idle
                # zero messages do not acquire the semantic lease.
                if was_active:
                    self._semantic_stop_until = now + rospy.Duration(
                        self.semantic_stop_hold_s
                    )
            else:
                self._semantic_active_until = now + rospy.Duration(
                    self.semantic_timeout_s
                )
                self._semantic_stop_until = rospy.Time(0)

    def _publish(self, _event) -> None:
        now = rospy.Time.now()
        with self._lock:
            semantic_age = (now - self._semantic_at).to_sec()
            move_base_age = (now - self._move_base_at).to_sec()
            semantic_active = (
                now <= self._semantic_active_until
                and semantic_age <= self.semantic_timeout_s
            )
            semantic_stopping = now <= self._semantic_stop_until

            if now <= self._explicit_stop_until:
                selected = Twist()
                source = "explicit_stop"
            elif semantic_active or semantic_stopping:
                selected = self._semantic
                source = "semantic"
            elif move_base_age <= self.move_base_timeout_s:
                selected = self._move_base
                source = "move_base"
            else:
                selected = Twist()
                source = "stale_stop"
            # Serialize publish with stop receipts; otherwise a timer could
            # snapshot forward velocity, then publish it after a stop arrived.
            self.publisher.publish(selected)

        if source != self._last_source:
            rospy.loginfo("velocity mux source: %s", source)
            self._last_source = source


if __name__ == "__main__":
    rospy.init_node("physical_velocity_command_mux")
    VelocityCommandMux()
    rospy.spin()
