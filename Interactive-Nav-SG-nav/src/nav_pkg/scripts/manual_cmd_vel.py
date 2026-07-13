#!/usr/bin/env python3
"""Keyboard teleop for MolmoSpaces ROS bridge mapping debug."""

import inspect
import os
import select
import sys
import termios
import tty
from typing import Optional


def patch_rosgraph_logger_for_py311() -> None:
    """Patch ROS Noetic logging when running under Python 3.11."""
    try:
        import rosgraph.roslogging as roslogging
    except Exception:
        return

    if not hasattr(roslogging, "RospyLogger"):
        return

    def _safe_find_caller(self, *args, **kwargs):
        file_name, lineno, func_name = super(roslogging.RospyLogger, self).findCaller(
            *args, **kwargs
        )[:3]
        file_name = os.path.normcase(file_name)

        f = inspect.currentframe()
        if f is not None:
            f = f.f_back
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename == file_name and f.f_lineno == lineno and co.co_name == func_name:
                break
            if f.f_back:
                f = f.f_back
            else:
                break

        if f is None or not hasattr(f, "f_code"):
            if sys.version_info > (3, 2):
                return file_name, lineno, func_name, None
            return file_name, lineno, func_name

        if f.f_back and f.f_code and f.f_code.co_name == "_base_logger":
            f = f.f_back
            if f.f_back:
                f = f.f_back
        co = f.f_code
        func_name2 = co.co_name
        try:
            class_name = f.f_locals["self"].__class__.__name__
            func_name2 = f"{class_name}.{func_name2}"
        except KeyError:
            pass

        if sys.version_info > (3, 2):
            return co.co_filename, f.f_lineno, func_name2, None
        return co.co_filename, f.f_lineno, func_name2

    roslogging.RospyLogger.findCaller = _safe_find_caller


patch_rosgraph_logger_for_py311()

import rospy
from geometry_msgs.msg import TwistStamped


HELP_TEXT = """
MolmoSpaces manual mapping control

  w: forward
  s: backward
  a: turn left in place
  d: turn right in place
  space: stop
  + / -: adjust speed
  q: quit
"""


class ManualCmdVel:
    def __init__(self) -> None:
        rospy.init_node("manual_cmd_vel")
        self.topic = rospy.get_param("~topic", "/cmd_vel_stamped")
        self.linear_speed = float(rospy.get_param("~linear_speed", 0.4))
        self.angular_speed = float(rospy.get_param("~angular_speed", 0.8))
        self.publish_rate = float(rospy.get_param("~publish_rate", 10.0))
        self.speed_step = float(rospy.get_param("~speed_step", 0.1))

        self.publisher = rospy.Publisher(self.topic, TwistStamped, queue_size=1)
        self.linear_x = 0.0
        self.angular_z = 0.0
        self._old_terminal_settings = None

    def _set_motion_from_key(self, key: str) -> bool:
        if key == "w":
            self.linear_x = self.linear_speed
            self.angular_z = 0.0
            self._print_status("forward")
        elif key == "s":
            self.linear_x = -self.linear_speed
            self.angular_z = 0.0
            self._print_status("backward")
        elif key == "a":
            self.linear_x = 0.0
            self.angular_z = self.angular_speed
            self._print_status("turn left")
        elif key == "d":
            self.linear_x = 0.0
            self.angular_z = -self.angular_speed
            self._print_status("turn right")
        elif key == " ":
            self.stop()
            self._print_status("stop")
        elif key in ("+", "="):
            self.linear_speed += self.speed_step
            self.angular_speed += self.speed_step
            self._print_status("speed up")
        elif key in ("-", "_"):
            self.linear_speed = max(self.speed_step, self.linear_speed - self.speed_step)
            self.angular_speed = max(self.speed_step, self.angular_speed - self.speed_step)
            self._print_status("speed down")
        elif key == "q" or key == "\x03":
            return False
        return True

    def _print_status(self, label: str) -> None:
        print(
            f"{label}: vx={self.linear_x:.2f} m/s, wz={self.angular_z:.2f} rad/s "
            f"(linear={self.linear_speed:.2f}, angular={self.angular_speed:.2f})",
            flush=True,
        )

    def publish(self) -> None:
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.twist.linear.x = self.linear_x
        msg.twist.angular.z = self.angular_z
        self.publisher.publish(msg)

    def stop(self) -> None:
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.publish()

    def _read_key_nonblocking(self) -> Optional[str]:
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None
        return sys.stdin.read(1)

    def run(self) -> None:
        print(HELP_TEXT, flush=True)
        self._old_terminal_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        rate = rospy.Rate(self.publish_rate)

        try:
            while not rospy.is_shutdown():
                key = self._read_key_nonblocking()
                if key is not None and not self._set_motion_from_key(key):
                    break
                self.publish()
                rate.sleep()
        finally:
            self.stop()
            if self._old_terminal_settings is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_terminal_settings)
            print("manual_cmd_vel stopped", flush=True)


if __name__ == "__main__":
    try:
        ManualCmdVel().run()
    except rospy.ROSInterruptException:
        pass
