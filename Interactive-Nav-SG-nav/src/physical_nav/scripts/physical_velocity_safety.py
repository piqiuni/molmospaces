#!/usr/bin/env python3
"""Package-local launcher for the physical velocity safety node."""

from pathlib import Path
import os
import sys

IMPLEMENTATION = (
    Path(__file__).resolve().parents[4]
    / "scripts" / "InteractiveNav" / "physical_nav" / "physical_velocity_safety.py"
)
sys.path.insert(0, str(IMPLEMENTATION.parent))

if __name__ == "__main__":
    os.execv(
        sys.executable,
        [sys.executable, str(IMPLEMENTATION), *sys.argv[1:]],
    )
