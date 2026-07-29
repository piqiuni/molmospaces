from __future__ import annotations

import math
from typing import Any, Iterable

from .frontier_core import FrontierCluster, GridSpec


def command_goal_xyyaw(command: dict[str, Any]) -> tuple[float, float, float] | None:
    """Return a finite dispatched frontier goal, if the command supplies one."""

    values = list(command.get("goal_xyyaw") or [])
    if len(values) < 2:
        return None
    try:
        x = float(values[0])
        y = float(values[1])
        yaw = float(values[2]) if len(values) > 2 else 0.0
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, yaw)):
        return None
    return x, y, yaw


def _finite_xy(value: Any) -> tuple[float, float] | None:
    values = list(value or [])
    if len(values) < 2:
        return None
    try:
        x, y = float(values[0]), float(values[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def preserved_frontier_cluster_from_command(
    command: dict[str, Any],
    *,
    grid_spec: GridSpec | None,
    robot_xy: tuple[float, float] | None,
) -> FrontierCluster | None:
    """Recreate an external reservation after map reclustering removes it.

    This object is bookkeeping only: semantic_behavior_executor remains the
    owner of make_plan and navigation safety preflight in external-control
    mode.  Keeping the dispatched goal here prevents a new scan from turning
    an in-flight exploration command into ``frontier_not_available``.
    """

    goal = command_goal_xyyaw(command)
    if goal is None:
        return None
    x, y, yaw = goal
    frontier_xy = _finite_xy(command.get("frontier_point")) or (x, y)
    cluster_id = str(
        command.get("cluster_id") or command.get("candidate_id") or "external_frontier"
    )
    if grid_spec is None:
        subgoal_cell = (0, 0)
        centroid_cell = (0.0, 0.0)
    else:
        subgoal_cell = grid_spec.world_to_grid(x, y)
        frontier_cell = grid_spec.world_to_grid(*frontier_xy)
        centroid_cell = (float(frontier_cell[0]), float(frontier_cell[1]))
    distance_to_robot = (
        math.hypot(x - float(robot_xy[0]), y - float(robot_xy[1]))
        if robot_xy is not None
        else 0.0
    )
    return FrontierCluster(
        cluster_id=cluster_id,
        cells=[],
        centroid_cell=centroid_cell,
        centroid_world=frontier_xy,
        subgoal_cell=subgoal_cell,
        subgoal_world=(x, y),
        subgoal_yaw=yaw,
        information_gain=0.0,
        distance_to_robot=distance_to_robot,
    )


def resolve_external_frontier_reservation(
    command: dict[str, Any],
    clusters: Iterable[FrontierCluster],
    *,
    grid_spec: GridSpec | None,
    robot_xy: tuple[float, float] | None,
) -> tuple[FrontierCluster | None, bool]:
    """Return the current cluster or a retained command goal after reclustering.

    The boolean is true only for the retained-goal path, making it observable
    in the reservation acknowledgement and recorder traces.
    """

    cluster_id = str(command.get("cluster_id") or "")
    cluster = next(
        (
            candidate
            for candidate in clusters
            if str(candidate.cluster_id) == cluster_id
        ),
        None,
    )
    if cluster is not None:
        return cluster, False
    retained = preserved_frontier_cluster_from_command(
        command,
        grid_spec=grid_spec,
        robot_xy=robot_xy,
    )
    return retained, retained is not None
