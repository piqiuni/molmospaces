#!/usr/bin/env python3
"""Package-local launcher for the system-Python physical ROS gateway.

The implementation lives with the non-ROS physical platform scripts so the
algorithm/web side can be used without a catkin checkout.  Keeping this tiny
executable in the ROS package also lets ``roslaunch`` locate it directly from
the source tree before an install step has been run.
"""

from pathlib import Path
import runpy


IMPLEMENTATION = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "InteractiveNav"
    / "physical_nav"
    / "physical_ros_gateway.py"
)


if __name__ == "__main__":
    runpy.run_path(str(IMPLEMENTATION), run_name="__main__")
