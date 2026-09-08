#!/usr/bin/env python3
"""Package-local launcher for the direct physical sensor ROS bridge."""

from pathlib import Path
import runpy


IMPLEMENTATION = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "InteractiveNav"
    / "physical_nav"
    / "physical_sensor_ros_bridge.py"
)


if __name__ == "__main__":
    runpy.run_path(str(IMPLEMENTATION), run_name="__main__")
