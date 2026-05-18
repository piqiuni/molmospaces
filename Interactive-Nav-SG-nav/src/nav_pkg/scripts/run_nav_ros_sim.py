#!/usr/bin/env python3
"""roslaunch wrapper for the MolmoSpaces nav ROS simulation script."""

import os
from pathlib import Path
import runpy
import sys


def strip_roslaunch_remaps(argv: list[str]) -> list[str]:
    """Remove ROS remapping args before forwarding to argparse-based scripts."""
    return [arg for arg in argv if not arg.startswith("__")]


def find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        sim_script = parent / "scripts" / "InteractiveNav" / "run_nav_ros_sim.py"
        if sim_script.exists():
            return parent
    raise RuntimeError("Could not find scripts/InteractiveNav/run_nav_ros_sim.py from wrapper path")


def main() -> None:
    repo_root = find_repo_root(Path(__file__).resolve())
    sim_script = repo_root / "scripts" / "InteractiveNav" / "run_nav_ros_sim.py"

    os.chdir(repo_root)
    sys.path.insert(0, str(sim_script.parent))
    sys.path.insert(0, str(repo_root))
    sys.argv = [str(sim_script), *strip_roslaunch_remaps(sys.argv[1:])]
    runpy.run_path(str(sim_script), run_name="__main__")


if __name__ == "__main__":
    main()
