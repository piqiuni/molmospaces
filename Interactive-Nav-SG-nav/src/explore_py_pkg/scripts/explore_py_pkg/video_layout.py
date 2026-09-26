"""Shared layouts and room-map background treatment for recorded videos."""

import cv2
import numpy as np
from explore_py_pkg.subgoal_overlay import SubgoalOverlay
from explore_py_pkg.debug_semantic_viz import candidate_color


PAPER_BACKGROUND = (218, 233, 244)  # BGR: light tan


def room_panel_background(panel, occupancy, room, *, room_mask=None, wall_mask=None):
    """Blend maps, keeping segmented rooms and measured walls visible."""
    if occupancy is not None:
        panel = cv2.addWeighted(occupancy, 0.72, panel, 0.28, 0.0)
    if room is not None:
        panel = cv2.addWeighted(room, 0.38, panel, 0.62, 0.0)
    if room_mask is not None:
        preserve = room_mask.copy()
        if wall_mask is not None:
            preserve |= wall_mask
        panel[~preserve] = PAPER_BACKGROUND
    return panel


def compose_video_panels(camera, occ, room, costmaps, spatial, topology, *, layout):
    if layout == "four_panel":
        return np.vstack([np.concatenate([camera, room], axis=1),
                          np.concatenate([spatial, topology], axis=1)])
    if layout == "six_panel":
        return np.vstack([np.concatenate([camera, occ, room], axis=1),
                          np.concatenate([costmaps, spatial, topology], axis=1)])
    raise ValueError(f"Unknown video layout: {layout}")


def draw_room_navigation_cues(
    panel, *, trajectory=(), global_plan=(), local_plan=(),
    candidates=(), live_goal=None, route_subgoals=(), interaction_goal=None,
    draw_paths=True, draw_markers=True,
):
    """Draw the OCC goal style over the room panel's existing viewport."""
    if draw_paths:
        for points, color, thickness in (
            (trajectory, (20, 118, 230), 2),
            (global_plan, (40, 190, 60), 3),
            (local_plan, (240, 150, 20), 3),
        ):
            points = list(points)
            if len(points) > 1:
                cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
    if not draw_markers:
        return
    for index, point, yaw in route_subgoals:
        SubgoalOverlay.draw_marker(panel, point, (210, 105, 35), radius=5, yaw=yaw)
        cv2.putText(panel, str(index), (point[0] + 7, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (35, 35, 35), 1, cv2.LINE_AA)
    if interaction_goal is not None:
        point, yaw = interaction_goal
        SubgoalOverlay.draw_marker(panel, point, (0, 140, 255), radius=6, selected=True)
        SubgoalOverlay.draw_direction(panel, point, yaw, 12, color=(0, 140, 255))
        cv2.putText(panel, "INTERACT", (point[0] + 8, point[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 105, 220), 1, cv2.LINE_AA)
    for point, behavior in candidates:
        SubgoalOverlay.draw_marker(panel, point, candidate_color(behavior), radius=4)
    if live_goal is not None:
        point, behavior, yaw = live_goal
        color = candidate_color(behavior)
        SubgoalOverlay.draw_marker(panel, point, color, radius=5, selected=True)
        SubgoalOverlay.draw_direction(panel, point, yaw, 12, color=color)
