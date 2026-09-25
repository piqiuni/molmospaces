"""Resolve the addon in both catkin source/devel and relocated install spaces."""

from pathlib import Path
import runpy
import sys


def runtime_root(package_root=None):
    if package_root is None:
        import rospkg
        package_root = rospkg.RosPack().get_path("physical_nav")
    # ROS1 recursively discovers executable/launch basenames, excluding dot
    # directories. Keep implementation files private so wrappers are unambiguous.
    root = Path(package_root) / ".runtime"
    if not (root / "physical_protocol.py").is_file():
        raise RuntimeError(f"physical_nav runtime missing from {root}; rebuild/install the addon")
    return root.resolve()


def run(entrypoint, package_root=None):
    root = runtime_root(package_root)
    if Path(entrypoint).name != entrypoint or not (root / entrypoint).is_file():
        raise ValueError(f"Unknown physical_nav entrypoint: {entrypoint}")
    # Legacy executables use sibling imports; limit compatibility to this entry.
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(root))
        runpy.run_path(str(root / entrypoint), run_name="__main__")
    finally:
        sys.path[:] = original_path
