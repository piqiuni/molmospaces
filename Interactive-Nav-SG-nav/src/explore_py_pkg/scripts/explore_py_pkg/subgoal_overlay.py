"""Canonical subgoal overlays shared by online and offline recorders."""

from __future__ import annotations

import math

import cv2
import numpy as np
from explore_py_pkg.debug_semantic_viz import candidate_color


class SubgoalOverlay:
    """Render the canonical offline-style subgoal marker and task banner."""

    @staticmethod
    def draw_marker(panel, center, color, *, radius, selected=False, yaw=None):
        radius = max(3, int(radius))
        cv2.circle(panel, center, radius, (18, 18, 18), -1, cv2.LINE_AA)
        cv2.circle(panel, center, max(2, radius - 1), tuple(int(v) for v in color), -1, cv2.LINE_AA)
        if selected and yaw is not None and math.isfinite(float(yaw)):
            SubgoalOverlay.draw_direction(panel, center, float(yaw), radius)

    @staticmethod
    def draw_direction(panel, center, yaw, length, color=(230, 30, 45)):
        heading = np.asarray([math.cos(float(yaw)), -math.sin(float(yaw))], dtype=np.float32)
        norm = float(np.linalg.norm(heading))
        if norm <= 1e-6:
            heading = np.asarray([1.0, 0.0], dtype=np.float32)
        else:
            heading /= norm
        length = max(6, int(length))
        start = np.asarray(center, dtype=np.float32) - heading * float(length * 0.18)
        end = np.asarray(center, dtype=np.float32) + heading * float(length)
        cv2.arrowedLine(panel, tuple(start.astype(np.int32)), tuple(end.astype(np.int32)), (18, 18, 18), 4, cv2.LINE_AA, tipLength=0.42)
        cv2.arrowedLine(panel, tuple(start.astype(np.int32)), tuple(end.astype(np.int32)), tuple(int(v) for v in color), 2, cv2.LINE_AA, tipLength=0.42)

    @staticmethod
    def draw_header(panel, target, behavior, name, *, box_width_px=None, background_alpha=1.0):
        box_width = min(panel.shape[1] - 4, max(120, int(box_width_px if box_width_px is not None else 460)))
        max_chars = max(13, int((box_width - 16) / 7.0))
        target, behavior, name = str(target or "-"), str(behavior or "-"), str(name or "-")
        def clipped(value, prefix):
            available = max(4, max_chars - len(prefix))
            return value if len(value) <= available else value[:max(1, available - 3)] + "..."
        overlay = panel.copy()
        cv2.rectangle(overlay, (4, 4), (box_width, 49), (255, 255, 255), -1)
        alpha = max(0.0, min(1.0, float(background_alpha)))
        if alpha >= 1.0:
            panel[:] = overlay
        elif alpha > 0.0:
            cv2.addWeighted(overlay, alpha, panel, 1.0 - alpha, 0.0, panel)
        for index, (prefix, value) in enumerate((("TASK TARGET: ", target), ("MODULE2: ", f"{behavior} {name}"))):
            cv2.putText(panel, prefix + clipped(value, prefix), (9, 20 + index * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1, cv2.LINE_AA)

    @staticmethod
    def render_candidate_sidebar(panel_size, candidates, selection, step_index=0):
        """Render the canonical ALL SUBGOALS sidebar used by both recorders."""
        width, height = panel_size
        panel = np.full((height, width, 3), (238, 242, 248), dtype=np.uint8)
        selected_id = str((selection or {}).get("candidate_id") or "")
        by_id = {}
        for raw in (candidates or []):
            item = dict(raw)
            cid = str(item.get("candidate_id") or "")
            if cid:
                by_id[cid] = item
        if selected_id:
            merged = dict(by_id.get(selected_id) or {})
            merged.update(selection or {})
            merged["candidate_id"] = selected_id
            by_id[selected_id] = merged
        priority = {"INTERACT": 0, "NAVIGATE": 1, "EXPLORE": 2}
        def score(item):
            for value in (item.get("score"), item.get("policy_score"), (item.get("features") or {}).get("score"), (item.get("features") or {}).get("pre_score")):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
            return 0.0
        rows = sorted(by_id.values(), key=lambda item: (priority.get(str(item.get("behavior_type") or "").upper(), 3), -score(item), str(item.get("candidate_id") or "")))
        cv2.rectangle(panel, (0, 0), (max(0, width - 1), max(0, height - 1)), (80, 80, 80), 1)
        cv2.putText(panel, "ALL SUBGOALS", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (35, 35, 35), 1, cv2.LINE_AA)
        if not rows:
            cv2.putText(panel, "-- none --", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (95, 95, 95), 1, cv2.LINE_AA)
            return panel
        row_h = max(8, min(22, max(1, height - 27) // len(rows)))
        font_scale = 0.25 if row_h >= 14 else 0.21
        for index, item in enumerate(rows):
            y1, y2 = 24 + index * row_h, min(height - 2, 24 + index * row_h + row_h - 2)
            if y1 >= height - 1:
                break
            cid = str(item.get("candidate_id") or "")
            behavior = str(item.get("behavior_type") or "EXPLORE").upper()
            color = candidate_color(behavior)
            selected = bool(selected_id and cid == selected_id)
            cv2.rectangle(panel, (3, y1), (width - 4, y2), (232, 246, 255) if selected else (255, 255, 255), -1)
            cv2.rectangle(panel, (3, y1), (width - 4, y2), (15, 15, 15), 2 if selected else 1)
            cv2.rectangle(panel, (4, y1 + 1), (10, max(y1 + 1, y2 - 1)), color, -1)
            target = str(item.get("target_name") or item.get("target_id") or cid or "-")
            label = f"{'>' if selected else ' '}{behavior[:3]} {target}"
            max_chars = max(5, int((width - 17) / max(3.5, 7.0 * font_scale)))
            cv2.putText(panel, label[:max_chars], (13, min(y2 - 2, y1 + max(7, row_h - 5))), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (25, 25, 25), 1, cv2.LINE_AA)
        return panel
