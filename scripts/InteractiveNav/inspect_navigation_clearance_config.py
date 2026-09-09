"""Print and validate the cross-module navigation-clearance contract.

This is intentionally a static pre-smoke check.  It does not alter ROS
parameters or tune a planner; it makes the physical footprint, costmap
inflation, frontier footprint check, and semantic recovery radii explicit so a
runtime smoke can be interpreted against the configuration that was launched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _number(payload: dict[str, Any], *keys: str) -> float:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Missing configuration key: {'.'.join(keys)}")
        current = current[key]
    value = float(current)
    if value < 0.0:
        raise ValueError(f"Configuration value must be non-negative: {'.'.join(keys)}")
    return value


def collect(repo_root: Path) -> dict[str, Any]:
    nav_dir = repo_root / "Interactive-Nav-SG-nav" / "src" / "nav_pkg" / "configs"
    explore_path = (
        repo_root
        / "Interactive-Nav-SG-nav"
        / "src"
        / "explore_py_pkg"
        / "config"
        / "explore_py.yaml"
    )
    semantic_path = (
        repo_root
        / "Interactive-Nav-SG-nav"
        / "src"
        / "semantic_decision_py_pkg"
        / "config"
        / "default.yaml"
    )
    common = _load_yaml(nav_dir / "costmap_common_params.yaml")
    local = _load_yaml(nav_dir / "local_costmap_params.yaml")
    global_costmap = _load_yaml(nav_dir / "global_costmap_params.yaml")
    explore = _load_yaml(explore_path)
    semantic = _load_yaml(semantic_path)

    physical_radius = _number(common, "robot_radius")
    local_inflation = _number(local, "local_costmap", "inflation_layer", "inflation_radius")
    global_inflation = _number(
        global_costmap, "global_costmap", "inflation_layer", "inflation_radius"
    )
    frontier_radius = _number(explore, "frontier", "robot_radius_m")
    frontier_margin = _number(explore, "frontier", "footprint_safety_margin_m")
    rear_radius = _number(semantic, "rear_goal_robot_radius_m")
    stuck_radius = _number(semantic, "stuck_recovery_robot_radius_m")
    effective_frontier_clearance = frontier_radius + frontier_margin

    epsilon = 1e-9
    violations: list[str] = []
    if physical_radius <= 0.0:
        violations.append("physical robot_radius must be positive")
    if global_inflation + epsilon < physical_radius:
        violations.append("global inflation is below the physical robot radius")
    if local_inflation + epsilon < physical_radius:
        violations.append("local inflation is below the physical robot radius")
    if abs(frontier_radius - physical_radius) > epsilon:
        violations.append("frontier robot_radius_m differs from costmap robot_radius")
    if effective_frontier_clearance + epsilon < local_inflation:
        violations.append("frontier footprint clearance is below local inflation")
    if abs(rear_radius - physical_radius) > epsilon:
        violations.append("semantic rear recovery radius differs from physical robot radius")
    if abs(stuck_radius - physical_radius) > epsilon:
        violations.append("semantic stuck recovery radius differs from physical robot radius")

    return {
        "ok": not violations,
        "violations": violations,
        "physical_robot_radius_m": physical_radius,
        "global_inflation_radius_m": global_inflation,
        "local_inflation_radius_m": local_inflation,
        "frontier_robot_radius_m": frontier_radius,
        "frontier_safety_margin_m": frontier_margin,
        "frontier_effective_clearance_m": effective_frontier_clearance,
        "semantic_rear_goal_radius_m": rear_radius,
        "semantic_stuck_recovery_radius_m": stuck_radius,
        "interpretation": (
            "global uses the physical footprint; local keeps an additional "
            f"{local_inflation - physical_radius:.2f} m collision buffer; "
            "frontier selection remains more conservative than local inflation"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    report = collect(args.repo_root.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
