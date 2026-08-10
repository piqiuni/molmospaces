#!/usr/bin/env python3
"""Pure PNG+JSON six-panel renderer used after a navigation run.

The recorder persists only raw grids, camera PNGs and state JSON.  This module
reconstructs the established debug renderer from those artifacts without ROS
or a live recorder process.  Keeping all coordinate conversion here makes the
offline frame a faithful, inspectable replay rather than a resized map image.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


_HELPER_DIR = (
    Path(__file__).resolve().parents[2]
    / "Interactive-Nav-SG-nav"
    / "src"
    / "explore_py_pkg"
    / "scripts"
)
if str(_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(_HELPER_DIR))

from explore_py_pkg.debug_semantic_viz import (  # noqa: E402
    candidate_color,
    portal_room_node_ids,
    topology_edge_style,
    topology_edge_visible,
    topology_hierarchy_layout,
    topology_node_style,
)


def _frame_name(value: object) -> str:
    return str(value or "").lstrip("/")


def _step4(value: int | float | None) -> str:
    return f"{int(value or 0):04d}"


def _yaw_from_origin(origin: dict | None) -> float:
    origin = origin or {}
    qx = float(origin.get("qx", 0.0) or 0.0)
    qy = float(origin.get("qy", 0.0) or 0.0)
    qz = float(origin.get("qz", 0.0) or 0.0)
    qw = float(origin.get("qw", 1.0) or 1.0)
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


@dataclass
class RawGrid:
    """A decoded OccupancyGrid image retaining its ROS world transform."""

    values: np.ndarray
    width: int
    height: int
    resolution: float
    frame_id: str
    origin_x: float
    origin_y: float
    origin_yaw: float

    @property
    def image_values(self) -> np.ndarray:
        # ROS OccupancyGrid row zero is the bottom row.  OpenCV image row zero
        # is the top row, therefore every raw PNG needs this exact flip.
        return np.flipud(self.values)

    def world_to_cell_unbounded(self, x: float, y: float) -> tuple[int, int] | None:
        if self.resolution <= 0.0:
            return None
        dx = float(x) - self.origin_x
        dy = float(y) - self.origin_y
        cos_yaw = math.cos(-self.origin_yaw)
        sin_yaw = math.sin(-self.origin_yaw)
        local_x = cos_yaw * dx - sin_yaw * dy
        local_y = sin_yaw * dx + cos_yaw * dy
        mx = int(math.floor(local_x / self.resolution))
        my = int(math.floor(local_y / self.resolution))
        return mx, my

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        cell = self.world_to_cell_unbounded(x, y)
        if cell is None:
            return None
        mx, my = cell
        if 0 <= mx < self.width and 0 <= my < self.height:
            return mx, my
        return None

    def world_from_cell(self, cell_x: float, cell_y: float) -> tuple[float, float]:
        cos_yaw = math.cos(self.origin_yaw)
        sin_yaw = math.sin(self.origin_yaw)
        local_x = float(cell_x) * self.resolution
        local_y = float(cell_y) * self.resolution
        return (
            self.origin_x + cos_yaw * local_x - sin_yaw * local_y,
            self.origin_y + sin_yaw * local_x + cos_yaw * local_y,
        )


def load_raw_grid(meta: dict | None, *, geometry: RawGrid | None = None) -> RawGrid | None:
    """Decode one lossless recorder PNG and its JSON geometry."""

    if not meta:
        return None
    image_path = Path(str(meta.get("image") or ""))
    encoded = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if encoded is None or encoded.ndim != 2:
        return None
    values = encoded.astype(np.int32) - int(meta.get("png_value_offset", 1) or 1)
    height = int(meta.get("height") or values.shape[0])
    width = int(meta.get("width") or values.shape[1])
    if values.shape != (height, width):
        return None
    if geometry is not None:
        return RawGrid(
            values=values,
            width=width,
            height=height,
            resolution=geometry.resolution,
            frame_id=geometry.frame_id,
            origin_x=geometry.origin_x,
            origin_y=geometry.origin_y,
            origin_yaw=geometry.origin_yaw,
        )
    origin = meta.get("origin") or {}
    return RawGrid(
        values=values,
        width=width,
        height=height,
        resolution=float(meta.get("resolution") or 0.0),
        frame_id=_frame_name(meta.get("frame_id")),
        origin_x=float(origin.get("x", 0.0) or 0.0),
        origin_y=float(origin.get("y", 0.0) or 0.0),
        origin_yaw=_yaw_from_origin(origin),
    )


def _receipt_number(receipt_id: object) -> int:
    try:
        return int(str(receipt_id or "").rsplit(":", 1)[-1])
    except ValueError:
        return -1


class GlobalCostmapReplay:
    """Apply saved OccupancyGridUpdate PNGs to the corresponding full grid."""

    def __init__(self, maps_by_id: dict[str, dict]) -> None:
        self.maps_by_id = maps_by_id
        self.updates = sorted(
            (
                record
                for record in maps_by_id.values()
                if str(record.get("stage") or "") == "global_costmap_update"
            ),
            key=lambda record: _receipt_number(record.get("receipt_id")),
        )
        self._base_receipt = ""
        self._grid: RawGrid | None = None
        self._applied_update = -1

    def grid_for(
        self,
        full_meta: dict | None,
        update_receipt: object,
        *,
        max_stamp_sec: float = 0.0,
    ) -> RawGrid | None:
        if not full_meta:
            return None
        full_receipt = str(full_meta.get("receipt_id") or "")
        if full_receipt != self._base_receipt:
            self._base_receipt = full_receipt
            self._grid = load_raw_grid(full_meta)
            self._applied_update = -1
        if self._grid is None:
            return None
        target = _receipt_number(update_receipt)
        if target < 0:
            return self._grid
        # A replay run is chronological.  Be robust if a caller seeks backwards
        # by rebuilding from the immutable full image.
        if target < self._applied_update:
            self._grid = load_raw_grid(full_meta)
            self._applied_update = -1
        for update in self.updates:
            index = _receipt_number(update.get("receipt_id"))
            if index <= self._applied_update:
                continue
            if index > target:
                break
            update_stamp = 0.0
            try:
                update_stamp = float(update.get("stamp_sec") or 0.0)
            except (TypeError, ValueError):
                pass
            # The selected receipt is normally already causal.  Keep this
            # second guard in the replay itself so an out-of-order manifest or
            # a legacy caller cannot accidentally apply a future patch.
            if max_stamp_sec > 0.0 and update_stamp > max_stamp_sec + 1e-6:
                continue
            patch = load_raw_grid(update, geometry=self._grid)
            if patch is None:
                continue
            x = int(update.get("x", 0) or 0)
            y = int(update.get("y", 0) or 0)
            h, w = patch.values.shape
            if x == 0 and y == 0 and (w, h) == (self._grid.width, self._grid.height):
                self._grid.values = patch.values
            elif x >= 0 and y >= 0 and x + w <= self._grid.width and y + h <= self._grid.height:
                self._grid.values[y : y + h, x : x + w] = patch.values
            self._applied_update = index
        return self._grid


@dataclass
class TransformSample:
    step_index: int
    x: float
    y: float
    yaw: float


class TransformResolver:
    """Offline equivalent of the recorder's map<-odom TF lookup."""

    def __init__(
        self,
        records: Iterable[TransformSample],
        *,
        map_frame: str,
        odom_frame: str,
    ) -> None:
        self.records = sorted(records, key=lambda item: item.step_index)
        self.map_frame = _frame_name(map_frame)
        self.odom_frame = _frame_name(odom_frame)

    @classmethod
    def from_csv(cls, path: Path, *, map_frame: str, odom_frame: str) -> "TransformResolver":
        records: list[TransformSample] = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    try:
                        records.append(
                            TransformSample(
                                step_index=int(row.get("step_id") or 0),
                                x=float(row.get("x") or 0.0),
                                y=float(row.get("y") or 0.0),
                                yaw=float(row.get("yaw") or 0.0),
                            )
                        )
                    except (TypeError, ValueError):
                        continue
        return cls(records, map_frame=map_frame, odom_frame=odom_frame)

    @classmethod
    def from_step_boundaries(
        cls,
        steps: Iterable[dict],
        *,
        map_frame: str,
        odom_frame: str,
        fallback_csv: Path | None = None,
    ) -> "TransformResolver":
        records: list[TransformSample] = []
        for index, step in enumerate(steps):
            payload = step.get("tf_map_from_odom") or {}
            try:
                if not payload:
                    continue
                records.append(
                    TransformSample(
                        step_index=int(step.get("step_index", index)),
                        x=float(payload.get("x") or 0.0),
                        y=float(payload.get("y") or 0.0),
                        yaw=float(payload.get("yaw") or 0.0),
                    )
                )
            except (TypeError, ValueError):
                continue
        if records or fallback_csv is None:
            return cls(records, map_frame=map_frame, odom_frame=odom_frame)
        return cls.from_csv(fallback_csv, map_frame=map_frame, odom_frame=odom_frame)

    def _sample(self, step_index: int) -> TransformSample:
        selected = TransformSample(step_index=0, x=0.0, y=0.0, yaw=0.0)
        for record in self.records:
            if record.step_index > step_index:
                break
            selected = record
        return selected

    def transform(
        self,
        x: float,
        y: float,
        yaw: float,
        source_frame: object,
        target_frame: object,
        step_index: int,
    ) -> tuple[float, float, float] | None:
        source = _frame_name(source_frame)
        target = _frame_name(target_frame)
        if not source or not target or source == target:
            return float(x), float(y), float(yaw)
        sample = self._sample(step_index)
        if source == self.odom_frame and target == self.map_frame:
            cos_yaw = math.cos(sample.yaw)
            sin_yaw = math.sin(sample.yaw)
            return (
                sample.x + cos_yaw * float(x) - sin_yaw * float(y),
                sample.y + sin_yaw * float(x) + cos_yaw * float(y),
                float(yaw) + sample.yaw,
            )
        if source == self.map_frame and target == self.odom_frame:
            dx = float(x) - sample.x
            dy = float(y) - sample.y
            cos_yaw = math.cos(-sample.yaw)
            sin_yaw = math.sin(-sample.yaw)
            return (
                cos_yaw * dx - sin_yaw * dy,
                sin_yaw * dx + cos_yaw * dy,
                float(yaw) - sample.yaw,
            )
        return None


# OpenCV uses BGR tuples.  Keep the colours as named constants because the
# replay is a debugging surface: a colour must retain one stable semantic
# meaning between runs.
RAW_FRONTIER_COLOR = (112, 36, 170)
UNSELECTED_EXPLORE_COLOR = (225, 185, 215)
UNSELECTED_EXPLORE_BORDER_COLOR = (178, 122, 170)
COSTMAP_SOFT_LIGHT_COLOR = (196, 248, 255)
COSTMAP_SOFT_DARK_COLOR = (65, 190, 255)
COSTMAP_INSCRIBED_COLOR = (25, 105, 255)
COSTMAP_LETHAL_COLOR = (30, 30, 150)


def _occupancy_base(grid: RawGrid) -> np.ndarray:
    raw_values = grid.values
    free = (raw_values >= 0) & (raw_values <= 20)
    unknown = raw_values < 0
    unknown_neighbor = np.zeros_like(unknown, dtype=bool)
    unknown_neighbor[:, 1:] |= unknown[:, :-1]
    unknown_neighbor[:, :-1] |= unknown[:, 1:]
    unknown_neighbor[1:, :] |= unknown[:-1, :]
    unknown_neighbor[:-1, :] |= unknown[1:, :]
    raw_frontier = free & unknown_neighbor
    values = np.flipud(raw_values)
    image = np.empty((grid.height, grid.width, 3), dtype=np.uint8)
    image[values < 0] = (178, 178, 178)
    image[(values >= 0) & (values <= 20)] = (248, 248, 245)
    image[(values > 20) & (values < 50)] = (118, 118, 118)
    image[values >= 50] = (28, 30, 32)
    image[np.flipud(raw_frontier)] = RAW_FRONTIER_COLOR
    return image


def _costmap_base(grid: RawGrid) -> np.ndarray:
    values = grid.image_values
    image = np.empty((grid.height, grid.width, 3), dtype=np.uint8)
    image[values < 0] = (178, 178, 178)
    image[values == 0] = (248, 248, 245)
    inflated = (values > 0) & (values < 99)
    if np.any(inflated):
        strength = values[inflated].astype(np.float32) / 98.0
        low = np.asarray(COSTMAP_SOFT_LIGHT_COLOR, dtype=np.float32)
        high = np.asarray(COSTMAP_SOFT_DARK_COLOR, dtype=np.float32)
        colors = low + (high - low) * strength[:, None]
        image[inflated] = np.clip(colors, 0, 255).astype(np.uint8)
    # Values 1--98 are traversable soft inflation, 99 is an inscribed-footprint
    # collision, and >=100 is lethal.  Use visibly separated yellow/orange/red
    # bands so a display cannot make an inflated cost look impassable.
    image[values == 99] = COSTMAP_INSCRIBED_COLOR
    image[values >= 100] = COSTMAP_LETHAL_COLOR
    return image


def _room_base(grid: RawGrid) -> np.ndarray:
    values = grid.image_values
    image = np.zeros((grid.height, grid.width, 3), dtype=np.uint8)
    palette = (
        (255, 185, 185),
        (185, 220, 255),
        (195, 245, 195),
        (245, 220, 170),
        (225, 195, 245),
        (175, 235, 230),
        (245, 195, 225),
        (220, 220, 170),
    )
    valid = values >= 0
    for room_id in np.unique(values[valid]) if np.any(valid) else []:
        image[values == int(room_id)] = palette[int(room_id) % len(palette)]
    return image


def _draw_panel_title(panel: np.ndarray, title: str, step_index: int) -> None:
    text = f"{title}  STEP={_step4(step_index)}"
    font_scale = 0.46 if panel.shape[1] < 700 else 0.58
    thickness = 1 if panel.shape[1] < 700 else 2
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x = max(6, panel.shape[1] - text_size[0] - 8)
    y = max(text_size[1] + 5, 22)
    cv2.putText(panel, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 245, 245), thickness + 2, cv2.LINE_AA)
    cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (25, 25, 25), thickness, cv2.LINE_AA)


def draw_map_snapshot_note(
    panel: np.ndarray,
    snapshot_meta: dict | None,
    *,
    display_stamp_sec: float = 0.0,
    selection_reason: str = "",
    y: int = 57,
) -> None:
    """Annotate a map which is intentionally older than the displayed frame.

    The replay must never borrow a receipt from the future just to make a map
    look up-to-date.  When a component has not published by the simulator-frame
    boundary, this compact label makes the resulting legitimate map lag visible
    instead of presenting it as a current-step map.
    """

    if not snapshot_meta:
        return
    try:
        snapshot_stamp = float(snapshot_meta.get("stamp_sec") or 0.0)
        display_stamp = float(display_stamp_sec or 0.0)
    except (TypeError, ValueError):
        return
    age_sec = display_stamp - snapshot_stamp if display_stamp and snapshot_stamp else 0.0
    reason = str(selection_reason or "")
    # A receipt selected through the causal fallback is worth surfacing even
    # when its timestamps round to the same video frame.
    if age_sec <= 0.05 and reason in {"requested", "latest_causal"}:
        return
    if age_sec > 0.0:
        label = f"MAP AS-OF {age_sec:.2f}s EARLIER"
    elif reason.startswith("requested_future"):
        label = "MAP AS-OF PRIOR CAUSAL RECEIPT"
    elif reason:
        label = "MAP SNAPSHOT SELECTED CAUSALLY"
    else:
        return
    font_scale = 0.31 if panel.shape[1] < 360 else 0.36
    thickness = 1
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x0, y0 = 6, max(15, int(y))
    x1 = min(panel.shape[1] - 5, x0 + text_size[0] + 10)
    overlay = panel.copy()
    cv2.rectangle(overlay, (x0 - 2, y0 - text_size[1] - 5), (x1, y0 + 4), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.78, panel, 0.22, 0.0, panel)
    cv2.putText(
        panel,
        label,
        (x0 + 2, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (55, 55, 115),
        thickness,
        cv2.LINE_AA,
    )


def _draw_costmap_legend(panel: np.ndarray) -> None:
    """Draw the cost semantics without conflating soft and collision costs."""

    legend = (
        ("SOFT 1-98", COSTMAP_SOFT_DARK_COLOR),
        ("INSCRIBED 99", COSTMAP_INSCRIBED_COLOR),
        ("LETHAL 100+", COSTMAP_LETHAL_COLOR),
    )
    height, width = panel.shape[:2]
    overlay = panel.copy()
    if width >= 390:
        x0, y0 = 6, height - 26
        cv2.rectangle(overlay, (x0 - 2, y0 - 3), (min(width - 4, 414), height - 3), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.82, panel, 0.18, 0.0, panel)
        x = x0 + 4
        for label, color in legend:
            cv2.rectangle(panel, (x, y0 + 2), (x + 12, y0 + 14), color, -1)
            cv2.putText(panel, label, (x + 17, y0 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (20, 20, 20), 1, cv2.LINE_AA)
            x += 132
        return
    # The two costmap panels are each 240 px wide in the standard layout.
    # A short stacked legend remains readable instead of clipping a horizontal
    # three-way legend beyond the panel edge.
    x0, y0 = 6, height - 48
    cv2.rectangle(overlay, (x0 - 2, y0 - 3), (min(width - 4, x0 + 116), height - 3), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.82, panel, 0.18, 0.0, panel)
    for index, (label, color) in enumerate(legend):
        baseline = y0 + 12 + index * 14
        cv2.rectangle(panel, (x0 + 3, baseline - 10), (x0 + 13, baseline), color, -1)
        cv2.putText(panel, label, (x0 + 18, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (20, 20, 20), 1, cv2.LINE_AA)


def _draw_occupancy_candidate_legend(
    panel: np.ndarray, live_behavior_type: str = ""
) -> None:
    """Explain raw frontiers, explore options, and the current live subgoal."""

    live_behavior = str(live_behavior_type or "").upper()
    if live_behavior not in {"EXPLORE", "NAVIGATE", "INTERACT"}:
        live_behavior = "NAVIGATE"
    legend = (
        ("RAW FRONTIER", RAW_FRONTIER_COLOR),
        ("EXPLORE OPTION", UNSELECTED_EXPLORE_COLOR),
        (f"LIVE {live_behavior}", candidate_color(live_behavior)),
    )
    height, width = panel.shape[:2]
    overlay = panel.copy()
    x0, y0 = 6, height - 25
    box_width = min(width - 4, 325)
    cv2.rectangle(overlay, (x0 - 2, y0 - 3), (box_width, height - 3), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.82, panel, 0.18, 0.0, panel)
    x = x0 + 4
    for label, color in legend:
        cv2.circle(panel, (x + 6, y0 + 8), 5, color, -1, cv2.LINE_AA)
        cv2.putText(panel, label, (x + 15, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (20, 20, 20), 1, cv2.LINE_AA)
        x += 104


def _nonnegative_int(value: object, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


def terminal_status_summary(step: dict) -> dict[str, object]:
    """Return a display-only summary of the recorded terminal progress.

    The renderer must not infer a new terminal condition.  It only surfaces
    fields already published by the candidate/decision streams.  A later
    recorder may include a completion-monitor snapshot; old recordings still
    show the known exhausted counts and the terminal-no-plan tracker when it is
    present in the decision trace.
    """

    candidates = step.get("semantic_candidates") or {}
    exploration = candidates.get("exploration_context") or {}
    trace = step.get("semantic_decision_trace") or {}
    terminal_no_plan = trace.get("terminal_no_plan_exit") or {}
    completion = (
        step.get("completion_status")
        or trace.get("completion_status")
        or candidates.get("completion_status")
        or {}
    )
    if not isinstance(completion, dict):
        completion = {}

    navigation_count = _nonnegative_int(
        exploration.get("navigation_frontier_count")
    )
    interaction_count = _nonnegative_int(
        exploration.get("interaction_frontier_count")
    )
    exhausted = bool(exploration.get("frontier_exhausted", False))
    has_context = bool(exploration)

    if bool(terminal_no_plan.get("armed", False)) or bool(
        terminal_no_plan.get("complete", False)
    ):
        detail = terminal_no_plan.get("detail") or {}
        elapsed = _nonnegative_int(detail.get("no_executable_elapsed_steps"))
        min_steps = _nonnegative_int(
            detail.get("no_executable_candidate_min_steps")
        )
        confirmations = _nonnegative_int(
            detail.get("no_executable_observation_confirmations")
        )
        required_confirmations = max(
            1,
            _nonnegative_int(
                detail.get("no_executable_candidate_confirmations_required"), 1
            ),
        )
        remaining_steps = max(0, min_steps - elapsed)
        remaining_confirmations = max(0, required_confirmations - confirmations)
        if bool(terminal_no_plan.get("complete", False)):
            label = "TERMINAL NO-PLAN EXIT CONFIRMED"
        else:
            label = (
                "TERMINAL NO-PLAN: "
                f"STEP {elapsed}/{min_steps} · OBS {confirmations}/{required_confirmations} "
                f"· REM {remaining_steps} STEP / {remaining_confirmations} OBS"
            )
        return {
            "label": label,
            "color": (55, 70, 225),
            "navigation_frontier_count": navigation_count,
            "interaction_frontier_count": interaction_count,
            "remaining_steps": remaining_steps,
            "remaining_confirmations": remaining_confirmations,
        }

    completion_config = completion.get("config") or {}
    confirmations = _nonnegative_int(
        completion.get(
            "frontier_confirmations",
            exploration.get("completion_confirmations", 0),
        )
    )
    required_confirmations = _nonnegative_int(
        completion_config.get(
            "frontier_confirmations",
            completion.get(
                "frontier_confirmations_required",
                exploration.get("completion_confirmations_required", 0),
            ),
        )
    )
    remaining_steps = _nonnegative_int(
        completion.get(
            "remaining_steps",
            exploration.get("completion_remaining_steps", 0),
        )
    )
    if bool(completion.get("requested", False)):
        label = "COMPLETION REQUESTED"
    elif exhausted:
        label = (
            f"FRONTIERS EXHAUSTED: NAV {navigation_count} · INTERACT {interaction_count}"
        )
        if required_confirmations:
            label += (
                f" · CONF {confirmations}/{required_confirmations}"
                f" · REM {max(0, required_confirmations - confirmations)} OBS"
            )
        if remaining_steps:
            label += f" / {remaining_steps} STEP"
    else:
        label = ""
    if not label and not has_context:
        return {}
    return {
        "label": label,
        "color": (95, 70, 190),
        "navigation_frontier_count": navigation_count,
        "interaction_frontier_count": interaction_count,
        "remaining_steps": remaining_steps,
        "remaining_confirmations": max(0, required_confirmations - confirmations),
    }


def draw_terminal_status(panel: np.ndarray, step: dict, *, baseline_y: int = 77) -> None:
    """Draw the recorded exhaustion/terminal progress without hiding the OCC."""

    summary = terminal_status_summary(step)
    label = str(summary.get("label") or "")
    if not label:
        return
    font_scale = 0.30 if panel.shape[1] < 360 else 0.33
    thickness = 1
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x0 = 6
    y0 = max(text_size[1] + 4, int(baseline_y))
    x1 = min(panel.shape[1] - 5, x0 + text_size[0] + 10)
    overlay = panel.copy()
    cv2.rectangle(
        overlay,
        (x0 - 2, y0 - text_size[1] - 5),
        (x1, y0 + 4),
        (255, 255, 255),
        -1,
    )
    cv2.addWeighted(overlay, 0.78, panel, 0.22, 0.0, panel)
    cv2.putText(
        panel,
        label,
        (x0 + 2, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        tuple(summary.get("color") or (55, 55, 170)),
        thickness,
        cv2.LINE_AA,
    )


def _draw_polyline(panel: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int], thickness: int) -> None:
    if len(points) >= 2:
        cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def _draw_faded_trajectory(
    panel: np.ndarray,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int,
    *,
    oldest_alpha: float = 0.14,
    newest_alpha: float = 0.95,
    buckets: int = 8,
) -> None:
    """Draw a chronological trail without hiding its older portions.

    Raw trajectories can contain one sample for nearly every simulator step.
    Rendering every segment with a separate alpha blend is unnecessarily slow,
    so consecutive segments are grouped into a small number of age buckets.
    The visual result remains a monotonic old-to-new fade while staying cheap
    enough for long offline replays.
    """

    if len(points) < 2:
        return
    compact = [points[0]]
    compact.extend(point for point in points[1:] if point != compact[-1])
    if len(compact) < 2:
        return
    segment_count = len(compact) - 1
    bucket_count = max(1, min(int(buckets), segment_count))
    for bucket_index in range(bucket_count):
        first_segment = int(math.floor(bucket_index * segment_count / bucket_count))
        last_segment = int(math.floor((bucket_index + 1) * segment_count / bucket_count))
        if last_segment <= first_segment:
            continue
        fraction = (bucket_index + 1) / bucket_count
        alpha = oldest_alpha + (newest_alpha - oldest_alpha) * fraction
        overlay = panel.copy()
        _draw_polyline(
            overlay,
            compact[first_segment : last_segment + 1],
            color,
            thickness,
        )
        cv2.addWeighted(overlay, float(alpha), panel, float(1.0 - alpha), 0.0, panel)


def zoom_panel(panel: np.ndarray, scale_factor: float) -> np.ndarray:
    """Match the established global-costmap diagnostic zoom."""
    scale_factor = max(1.0, float(scale_factor))
    if scale_factor <= 1.0 + 1e-6:
        return panel
    height, width = panel.shape[:2]
    enlarged = cv2.resize(
        panel,
        (max(width, int(round(width * scale_factor))), max(height, int(round(height * scale_factor)))),
        interpolation=cv2.INTER_NEAREST,
    )
    offset_x = max(0, (enlarged.shape[1] - width) // 2)
    offset_y = max(0, (enlarged.shape[0] - height) // 2)
    return enlarged[offset_y : offset_y + height, offset_x : offset_x + width].copy()


def zoom_world_bounds(
    bounds: tuple[float, float, float, float], scale_factor: float
) -> tuple[float, float, float, float]:
    """Zoom a semantic panel in world coordinates, keeping overlays aligned."""
    try:
        scale = float(scale_factor)
    except (TypeError, ValueError):
        scale = 1.0
    if not math.isfinite(scale) or scale <= 1.0 + 1e-6:
        return bounds
    min_x, min_y, max_x, max_y = bounds
    center_x, center_y = (min_x + max_x) * 0.5, (min_y + max_y) * 0.5
    half_width = max(1e-6, (max_x - min_x) * 0.5 / scale)
    half_height = max(1e-6, (max_y - min_y) * 0.5 / scale)
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def extend_world_bounds_lower(
    bounds: tuple[float, float, float, float] | None,
    lower_margin_m: float,
) -> tuple[float, float, float, float] | None:
    """Reserve extra map-frame space below the OCC panel without zoom settings.

    The extra interval is intentionally applied only to Panel 2's world bounds.
    It does not change the configured detail zoom for the room, semantic-XY, or
    costmap panels.
    """

    if bounds is None:
        return None
    try:
        margin = float(lower_margin_m)
    except (TypeError, ValueError):
        return bounds
    if not math.isfinite(margin) or margin <= 0.0:
        return bounds
    min_x, min_y, max_x, max_y = bounds
    return (min_x, min_y - margin, max_x, max_y)


def draw_task_subgoal_header(
    panel: np.ndarray,
    step: dict,
    *,
    box_width_px: int | None = None,
    background_alpha: float = 1.0,
) -> None:
    """Draw the OCC task/subgoal banner with configurable compact opacity."""

    selection = active_semantic_selection(step)
    candidates = step.get("semantic_candidates") or {}
    target = str((candidates.get("target_context") or {}).get("target_name") or "-")
    behavior = str(selection.get("behavior_type") or "-")
    name = str(selection.get("target_name") or selection.get("target_id") or selection.get("candidate_id") or "-")
    box_width = min(
        panel.shape[1] - 4,
        max(120, int(box_width_px if box_width_px is not None else 460)),
    )
    max_chars = max(13, int((box_width - 16) / 7.0))
    def clipped(value: str, prefix: str) -> str:
        available = max(4, max_chars - len(prefix))
        return value if len(value) <= available else value[: max(1, available - 3)] + "..."
    overlay = panel.copy()
    cv2.rectangle(overlay, (4, 4), (box_width, 49), (255, 255, 255), -1)
    alpha = max(0.0, min(1.0, float(background_alpha)))
    if alpha >= 1.0:
        panel[:] = overlay
    elif alpha > 0.0:
        cv2.addWeighted(overlay, alpha, panel, 1.0 - alpha, 0.0, panel)
    for index, line in enumerate((f"TASK TARGET: {target}", f"MODULE2 SUBGOAL: {behavior} {name}")):
        prefix = "TASK TARGET: " if index == 0 else "MODULE2: "
        value = target if index == 0 else f"{behavior} {name}"
        cv2.putText(
            panel,
            prefix + clipped(value, prefix),
            (9, 20 + index * 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )


def _draw_robot_arrow(panel: np.ndarray, center: tuple[int, int], yaw: float, length: int) -> None:
    cx, cy = center
    heading = np.asarray([math.cos(yaw), -math.sin(yaw)], dtype=np.float32)
    norm = float(np.linalg.norm(heading))
    heading = np.asarray([1.0, 0.0], dtype=np.float32) if norm <= 1e-6 else heading / norm
    perp = np.asarray([-heading[1], heading[0]], dtype=np.float32)
    tip = np.asarray([cx, cy], dtype=np.float32) + heading * float(length)
    back = np.asarray([cx, cy], dtype=np.float32) - heading * float(length * 0.55)
    left = back + perp * float(length * 0.45)
    right = back - perp * float(length * 0.45)
    points = np.asarray([tip, left, right], dtype=np.int32)
    cv2.fillConvexPoly(panel, points, (0, 88, 255), cv2.LINE_AA)
    cv2.polylines(panel, [points], True, (0, 35, 160), 2, cv2.LINE_AA)


def _draw_goal_arrow(panel: np.ndarray, center: tuple[int, int], yaw: float, length: int, color: tuple[int, int, int] = (230, 30, 45)) -> None:
    cx, cy = center
    heading = np.asarray([math.cos(yaw), -math.sin(yaw)], dtype=np.float32)
    norm = float(np.linalg.norm(heading))
    heading = np.asarray([1.0, 0.0], dtype=np.float32) if norm <= 1e-6 else heading / norm
    start = np.asarray([cx, cy], dtype=np.float32) - heading * float(length * 0.45)
    end = np.asarray([cx, cy], dtype=np.float32) + heading * float(length)
    cv2.arrowedLine(panel, tuple(start.astype(np.int32)), tuple(end.astype(np.int32)), color, 4, cv2.LINE_AA, tipLength=0.45)
    cv2.circle(panel, (cx, cy), max(4, length // 4), color, -1, cv2.LINE_AA)


def _selection_target_id(selection: dict | None) -> str:
    selection = selection or {}
    if selection.get("active") is False:
        return ""
    return str(selection.get("target_id") or selection.get("object_id") or selection.get("candidate_id") or "")


_CANONICAL_ID_PREFIXES = (
    "interaction_",
    "target_",
    "container_",
    "object_",
    "instance_",
    "portal_",
    "doorframe_",
)
_CANONICAL_CONTAINER_ALIASES = (
    ("chest_of_drawers", "drawer"),
    ("chestofdrawers", "drawer"),
    ("chest_drawers", "drawer"),
    ("dresser", "drawer"),
)


def canonical_object_id_variants(value: object) -> set[str]:
    """Return stable ID aliases shared by recorder selection and graph nodes.

    Runtime graph IDs can carry ``container_``/``object_``/``portal_`` source
    prefixes while the executor selection carries a candidate or simulator
    object ID.  Rendering must compare the semantic identity, not the producer
    prefix.  The alias normalization is intentionally identifier-only; it does
    not use GT class metadata to manufacture a match.
    """

    raw = str(value or "").strip().casefold()
    if not raw:
        return set()
    candidates = {raw}
    # Candidate IDs commonly look like interaction:<object>:open.  Keep the
    # object segment as an alias while retaining the full token for diagnostics.
    parts = [part.strip() for part in raw.split(":") if part.strip()]
    if len(parts) >= 2 and parts[0] in {"interaction", "target", "candidate"}:
        candidates.add(parts[1])
    variants: set[str] = set()
    for candidate in candidates:
        token = candidate.replace("-", "_").replace(" ", "_")
        token = "_".join(part for part in token.split("_") if part)
        variants.add(token)
        stripped = token
        changed = True
        while changed:
            changed = False
            for prefix in _CANONICAL_ID_PREFIXES:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix) :]
                    changed = True
                    break
        for source, replacement in _CANONICAL_CONTAINER_ALIASES:
            stripped = stripped.replace(source, replacement)
        stripped = "_".join(part for part in stripped.split("_") if part)
        if stripped:
            variants.add(stripped)
    return {variant for variant in variants if variant}


def selection_target_ids(selection: dict | None) -> set[str]:
    """Collect canonical IDs for the live selection and its interaction command."""

    selection = selection or {}
    if selection.get("active") is False:
        return set()
    values = [
        selection.get("target_id"),
        selection.get("object_id"),
        selection.get("candidate_id"),
        (selection.get("interaction_command") or {}).get("node_id"),
        (selection.get("interaction_command") or {}).get("object_id"),
    ]
    identifiers: set[str] = set()
    for value in values:
        identifiers.update(canonical_object_id_variants(value))
    # A name is a last-resort fallback for old recorder snapshots that carried
    # no object/node ID at all.  Prefer an exact ID whenever one exists.
    if not identifiers:
        identifiers.update(canonical_object_id_variants(selection.get("target_name")))
    return identifiers


def _node_selection_ids(node: dict) -> set[str]:
    attributes = node.get("attributes") or {}
    values = (
        attributes.get("object_id"),
        attributes.get("source_object_name"),
        attributes.get("instance_id"),
        node.get("object_id"),
        node.get("id"),
        node.get("name"),
    )
    identifiers: set[str] = set()
    for value in values:
        identifiers.update(canonical_object_id_variants(value))
    return identifiers


def node_matches_selection(node: dict, target_ids: set[str]) -> bool:
    """Match a graph node against canonical live-selection IDs."""

    return bool(target_ids and target_ids.intersection(_node_selection_ids(node)))


def _selection_revision(record: dict | None) -> str:
    """Read any supported immutable candidate-geometry revision field."""

    record = record or {}
    for key in (
        "selected_candidate_revision",
        "candidate_revision",
        "geometry_revision",
        "goal_revision",
        "revision",
    ):
        value = record.get(key)
        if value not in {None, ""}:
            return str(value)
    return ""


def candidate_matches_canonical_selection(
    selection: dict | None,
    candidate: dict | None,
    *,
    position_tolerance_m: float = 0.03,
    yaw_tolerance_rad: float = math.radians(5.0),
) -> bool:
    """Return whether a candidate snapshot represents the live selection.

    Candidate lists are asynchronous diagnostics.  An unchanged string ID is
    not enough: a frontier can be re-centred after mapping updates.  The live
    selection is the replay authority, and an older candidate may only receive
    selected styling when its immutable revision (when available) and geometry
    agree with that selection.
    """

    selection = selection or {}
    candidate = candidate or {}
    selected_id = str(selection.get("candidate_id") or "")
    if not selected_id or selected_id != str(candidate.get("candidate_id") or ""):
        return False
    selected_revision = _selection_revision(selection)
    candidate_revision = _selection_revision(candidate)
    if selected_revision and candidate_revision and selected_revision != candidate_revision:
        return False
    selected_goal = list(selection.get("goal_xyyaw") or [])
    candidate_goal = list(candidate.get("goal_xyyaw") or [])
    if len(selected_goal) < 2 or len(candidate_goal) < 2:
        return False
    try:
        position_error = math.hypot(
            float(selected_goal[0]) - float(candidate_goal[0]),
            float(selected_goal[1]) - float(candidate_goal[1]),
        )
    except (TypeError, ValueError):
        return False
    if position_error > float(position_tolerance_m):
        return False
    if len(selected_goal) < 3 or len(candidate_goal) < 3:
        return True
    try:
        yaw_error = math.atan2(
            math.sin(float(selected_goal[2]) - float(candidate_goal[2])),
            math.cos(float(selected_goal[2]) - float(candidate_goal[2])),
        )
    except (TypeError, ValueError):
        return False
    return abs(yaw_error) <= float(yaw_tolerance_rad)


def active_semantic_selection(step: dict) -> dict:
    """Return only a live selection, with executor-selected fallback geometry.

    A selection is latched for recorder discovery, so a terminal reset is
    explicit rather than inferred from an empty candidate stream.  During an
    active INTERACT approach the executor may choose a safe fallback pose; use
    that same pose in offline overlays rather than the decision-time primary.
    """

    raw_selection = step.get("semantic_selection") or {}
    if raw_selection.get("active") is False:
        return {}
    selection = dict(raw_selection)
    execution = step.get("semantic_execution_state") or {}
    if (
        str(execution.get("candidate_id") or "")
        and str(execution.get("candidate_id") or "")
        == str(selection.get("candidate_id") or "")
        and str(execution.get("state") or "")
        in {"APPROACH_INTERACTION", "WAIT_FOR_DRAWER_SCAN", "INTERACTING", "VERIFYING"}
    ):
        effective_goal = list(execution.get("effective_goal_xyyaw") or [])
        if len(effective_goal) >= 2:
            selection["goal_xyyaw"] = effective_goal
            selection["effective_goal_xyyaw"] = effective_goal
    return selection


def _node_ids(node: dict) -> set[str]:
    attributes = node.get("attributes") or {}
    return {
        str(value)
        for value in (
            attributes.get("object_id"),
            attributes.get("source_object_name"),
            attributes.get("instance_id"),
            node.get("id"),
        )
        if value
    }


def _node_observed(node: dict, observed_ids: set[str]) -> bool:
    if str(node.get("type") or "") == "room":
        return bool((node.get("attributes") or {}).get("active", True))
    return bool(_node_ids(node).intersection(observed_ids))


def _node_xy(node: dict) -> tuple[float, float] | None:
    values = node.get("aabb_center") or node.get("centroid") or []
    if len(values) < 2:
        return None
    return float(values[0]), float(values[1])


def _short_node_id(node: dict) -> str:
    attributes = node.get("attributes") or {}
    token = str(attributes.get("object_id") or attributes.get("source_object_name") or attributes.get("instance_id") or node.get("id") or "")
    suffix = token.rsplit("_", 1)[-1]
    try:
        return f"#{int(suffix)}"
    except ValueError:
        return suffix[-6:]


def _node_label(node: dict) -> str:
    if str(node.get("type") or "") != "room":
        return str(node.get("label") or node.get("type") or "object")
    value = str((node.get("attributes") or {}).get("room_attribute") or "unknown").strip()
    if value == "livingroom":
        return "living room"
    return f"{value} room" if value != "unknown" else "unknown room"


def _node_color(node: dict) -> tuple[int, int, int]:
    node_type = str(node.get("type") or "object")
    if node_type == "room":
        return (205, 225, 245)
    if node_type == "portal":
        state = str((node.get("interaction") or {}).get("state") or "unknown").casefold()
        if state in {"blocked", "unsupported"}:
            return (175, 45, 185)
        if state == "static_open":
            return (195, 175, 35)
        if state == "static_closed":
            return (150, 95, 105)
        if state == "static":
            return (125, 125, 125)
        return (50, 190, 70) if state in {"open", "ajar"} else (235, 70, 55) if state == "closed" else (235, 175, 45)
    if node_type == "container":
        return (175, 75, 220)
    if node_type == "support":
        return (50, 125, 220)
    return (30, 190, 195) if node.get("is_currently_visible") else (145, 145, 145)


def _bounded_nodes(
    graph: dict, observed_ids: set[str], target_ids: set[str], limit: int = 96
) -> list[dict]:
    # The current interaction target remains a diagnostic overlay even when a
    # live selection has just moved it out of the camera's observed-ID set.
    # Otherwise a remote drawer can be selected correctly but disappear from
    # the replay panel before its canonical INTERACT highlight is drawn.
    observed = [
        node
        for node in graph.get("nodes") or []
        if _node_observed(node, observed_ids)
        or node_matches_selection(node, target_ids)
    ]
    rooms = [node for node in observed if str(node.get("type") or "") == "room"]
    others = [node for node in observed if str(node.get("type") or "") != "room"]
    if len(others) <= limit:
        return rooms + others
    others.sort(
        key=lambda node: (
            0 if node_matches_selection(node, target_ids) else 1,
            0 if bool(node.get("is_currently_visible")) else 1,
            0 if str(node.get("type") or "") == "portal" else 1,
            0 if str(node.get("type") or "") == "container" else 1,
            str(node.get("id") or ""),
        )
    )
    return rooms + others[:limit]


def known_world_bounds(grid: RawGrid, margin_m: float = 2.5) -> tuple[float, float, float, float] | None:
    known = grid.values >= 0
    rows = np.flatnonzero(np.any(known, axis=1))
    cols = np.flatnonzero(np.any(known, axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    min_x, max_x = int(np.min(cols)), int(np.max(cols) + 1)
    min_y, max_y = int(np.min(rows)), int(np.max(rows) + 1)
    corners = [
        grid.world_from_cell(min_x, min_y),
        grid.world_from_cell(max_x, min_y),
        grid.world_from_cell(min_x, max_y),
        grid.world_from_cell(max_x, max_y),
    ]
    return (
        min(point[0] for point in corners) - margin_m,
        min(point[1] for point in corners) - margin_m,
        max(point[0] for point in corners) + margin_m,
        max(point[1] for point in corners) + margin_m,
    )


class OfflineSixPanelRenderer:
    """Stateful render replay sharing crop and transform rules with the old video."""

    def __init__(self, *, transforms: TransformResolver) -> None:
        self.transforms = transforms

    def _world_to_image_px(self, grid: RawGrid, values: tuple[float, float, float] | None) -> tuple[int, int] | None:
        if values is None:
            return None
        cell = grid.world_to_cell(values[0], values[1])
        return None if cell is None else (cell[0], grid.height - 1 - cell[1])

    def _transform(self, values: tuple[float, float, float] | list[float] | None, source: str, target: str, step: int) -> tuple[float, float, float] | None:
        if values is None or len(values) < 2:
            return None
        return self.transforms.transform(
            float(values[0]), float(values[1]), float(values[2]) if len(values) > 2 else 0.0,
            source, target, step,
        )

    def _plan_poses(self, plan: dict | None, grid: RawGrid, step: int) -> list[tuple[float, float, float]]:
        result = []
        for pose in (plan or {}).get("poses") or []:
            converted = self._transform(pose, str((plan or {}).get("frame_id") or ""), grid.frame_id, step)
            if converted is None:
                return []
            result.append(converted)
        return result

    def render_map_panel(
        self,
        grid: RawGrid | None,
        panel_size: tuple[int, int],
        step: dict,
        step_index: int,
        *,
        title: str,
        kind: str,
        world_bounds: tuple[float, float, float, float] | None = None,
        draw_global_plan: bool = True,
        draw_local_global_plan: bool = False,
        draw_local_plan: bool = True,
        draw_frontiers: bool = True,
        draw_semantic_candidates: bool = False,
        draw_route_plan: bool = False,
        episode_trajectory: list[tuple[float, float, float, float]] | None = None,
        view_scale: float = 1.0,
        snapshot_meta: dict | None = None,
        display_stamp_sec: float = 0.0,
        snapshot_selection_reason: str = "",
    ) -> np.ndarray:
        width, height = panel_size
        if grid is None:
            panel = np.full((height, width, 3), 235, dtype=np.uint8)
            cv2.putText(panel, f"NO {title} YET", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2, cv2.LINE_AA)
            return panel
        base = _costmap_base(grid) if kind == "costmap" else _occupancy_base(grid)
        pose = self._transform(step.get("pose"), self.transforms.odom_frame, grid.frame_id, step_index)
        raw_selection = step.get("semantic_selection") or {}
        selection = active_semantic_selection(step)
        goal_values = list(selection.get("goal_xyyaw") or [])
        active_goal = (
            None
            if raw_selection.get("active") is False
            else goal_values if len(goal_values) >= 2 else step.get("active_goal")
        )
        goal_yaw = float(goal_values[2]) if len(goal_values) > 2 else float(step.get("active_goal_yaw") or 0.0)
        goal = self._transform(active_goal, self.transforms.map_frame, grid.frame_id, step_index)
        trajectory_source = (
            episode_trajectory
            if episode_trajectory is not None
            else step.get("trajectory") or []
        )
        trajectory = [
            converted
            for raw in trajectory_source
            if len(raw) >= 4
            for converted in [self._transform((raw[1], raw[2], raw[3]), self.transforms.odom_frame, grid.frame_id, step_index)]
            if converted is not None
        ]
        global_plan = self._plan_poses(step.get("global_plan"), grid, step_index) if draw_global_plan else []
        local_global_plan = self._plan_poses(step.get("local_global_plan"), grid, step_index) if draw_local_global_plan else []
        local_plan = self._plan_poses(step.get("local_plan"), grid, step_index) if draw_local_plan else []
        if pose is not None:
            def prune(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
                if not points:
                    return points
                nearest = min(range(len(points)), key=lambda index: math.hypot(points[index][0] - pose[0], points[index][1] - pose[1]))
                return [pose] + points[nearest:]
            global_plan, local_global_plan, local_plan = prune(global_plan), prune(local_global_plan), prune(local_plan)

        def bounds_from_world(value: tuple[float, float, float, float]) -> tuple[int, int, int, int] | None:
            corners = [
                self._transform((value[0], value[1], 0.0), self.transforms.map_frame, grid.frame_id, step_index),
                self._transform((value[2], value[1], 0.0), self.transforms.map_frame, grid.frame_id, step_index),
                self._transform((value[0], value[3], 0.0), self.transforms.map_frame, grid.frame_id, step_index),
                self._transform((value[2], value[3], 0.0), self.transforms.map_frame, grid.frame_id, step_index),
            ]
            pixels = []
            for point in corners:
                if point is None:
                    continue
                cell = grid.world_to_cell_unbounded(point[0], point[1])
                if cell is not None:
                    pixels.append((cell[0], grid.height - 1 - cell[1]))
            if not pixels:
                return None
            xs, ys = zip(*pixels)
            return (
                max(0, min(grid.width - 1, min(xs))),
                max(0, min(grid.height - 1, min(ys))),
                max(0, min(grid.width - 1, max(xs))),
                max(0, min(grid.height - 1, max(ys))),
            )

        trajectory_pixels = [
            pixel
            for point in trajectory
            for pixel in [self._world_to_image_px(grid, point)]
            if pixel is not None
        ]
        view_bounds = (
            zoom_world_bounds(world_bounds, view_scale)
            if world_bounds is not None
            else None
        )
        crop = bounds_from_world(view_bounds) if view_bounds is not None else None
        # The established OCC/global bounds track known map extents, while the
        # episode trajectory can begin outside a newly cropped local extent.
        # Include every still-representable historic point so offline replay
        # never silently drops early path history.
        if crop is not None and trajectory_pixels:
            margin = max(8, int(math.ceil(4.5 / max(grid.resolution, 1e-6))))
            xs, ys = zip(*trajectory_pixels)
            crop = (
                max(0, min(crop[0], min(xs) - margin)),
                max(0, min(crop[1], min(ys) - margin)),
                min(grid.width - 1, max(crop[2], max(xs) + margin)),
                min(grid.height - 1, max(crop[3], max(ys) + margin)),
            )
        if crop is not None and (crop[2] <= crop[0] or crop[3] <= crop[1]):
            # A transformed global-map viewport can lie wholly outside a
            # smaller grid. Use the live overlays below rather than producing
            # a blank panel because the requested bounds have no local area.
            crop = None
        if crop is None:
            pixels = [self._world_to_image_px(grid, value) for value in [pose, goal, *trajectory, *global_plan, *local_global_plan, *local_plan]]
            pixels = [pixel for pixel in pixels if pixel is not None]
            if pixels:
                margin = max(8, int(math.ceil(4.5 / max(grid.resolution, 1e-6))))
                xs, ys = zip(*pixels)
                crop = (max(0, min(xs) - margin), max(0, min(ys) - margin), min(grid.width - 1, max(xs) + margin), min(grid.height - 1, max(ys) + margin))
            else:
                crop = (0, 0, grid.width - 1, grid.height - 1)
        min_x, min_y, max_x, max_y = crop
        if max_x <= min_x or max_y <= min_y:
            return np.full((height, width, 3), 235, dtype=np.uint8)
        image_crop = base[min_y : max_y + 1, min_x : max_x + 1]
        crop_h, crop_w = image_crop.shape[:2]
        scale = min(width / max(crop_w, 1), height / max(crop_h, 1))
        scaled_w, scaled_h = max(1, int(round(crop_w * scale))), max(1, int(round(crop_h * scale)))
        panel = np.full((height, width, 3), 235, dtype=np.uint8)
        offset_x, offset_y = (width - scaled_w) // 2, (height - scaled_h) // 2
        panel[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w] = cv2.resize(image_crop, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

        def to_panel(point: tuple[float, float, float] | None) -> tuple[int, int] | None:
            pixel = self._world_to_image_px(grid, point)
            if pixel is None or not (min_x <= pixel[0] <= max_x and min_y <= pixel[1] <= max_y):
                return None
            return int(round(offset_x + (pixel[0] - min_x) * scale)), int(round(offset_y + (pixel[1] - min_y) * scale))

        if draw_frontiers and goal is not None and kind != "costmap":
            free = (grid.values >= 0) & (grid.values <= 20)
            unknown = grid.values < 0
            neighbor = np.zeros_like(unknown, dtype=bool)
            neighbor[:, 1:] |= unknown[:, :-1]
            neighbor[:, :-1] |= unknown[:, 1:]
            neighbor[1:, :] |= unknown[:-1, :]
            neighbor[:-1, :] |= unknown[1:, :]
            frontier = free & neighbor
            goal_cell = grid.world_to_cell(goal[0], goal[1])
            if goal_cell is not None:
                radius = max(1, int(math.ceil(1.0 / max(grid.resolution, 1e-6))))
                gx, gy = goal_cell
                for cy in range(max(0, gy - radius), min(grid.height, gy + radius + 1)):
                    for cx in range(max(0, gx - radius), min(grid.width, gx + radius + 1)):
                        if not frontier[cy, cx]:
                            continue
                        point = (cx, grid.height - 1 - cy)
                        if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
                            cv2.circle(panel, (int(round(offset_x + (point[0] - min_x) * scale)), int(round(offset_y + (point[1] - min_y) * scale))), max(1, int(round(2.0 * max(scale, 1.0)))), (112, 36, 170), -1, cv2.LINE_AA)
        _draw_faded_trajectory(
            panel,
            [point for item in trajectory if (point := to_panel(item)) is not None],
            (20, 118, 230),
            3,
        )
        if pose is not None and (robot_px := to_panel(pose)) is not None:
            _draw_robot_arrow(panel, robot_px, pose[2], max(9, int(9 * scale)))
        if draw_global_plan:
            _draw_polyline(panel, [point for item in global_plan if (point := to_panel(item)) is not None], (40, 190, 60), 3)
        if draw_local_global_plan:
            _draw_polyline(panel, [point for item in local_global_plan if (point := to_panel(item)) is not None], (40, 190, 60), 3)
        if draw_local_plan:
            _draw_polyline(panel, [point for item in local_plan if (point := to_panel(item)) is not None], (240, 150, 20), 3)
        # Candidate lists are diagnostic dots. The live semantic selection gets
        # the only goal arrow, so every map panel has one canonical command.
        selected_id = str(selection.get("candidate_id") or "")
        selected_candidate_stale = False
        if draw_semantic_candidates:
            for candidate in (step.get("semantic_candidates") or {}).get("candidates") or []:
                values = list(candidate.get("goal_xyyaw") or [])
                candidate_point = self._transform(values, self.transforms.map_frame, grid.frame_id, step_index)
                candidate_px = to_panel(candidate_point)
                behavior_type = str(candidate.get("behavior_type") or "EXPLORE").upper()
                color = candidate_color(behavior_type)
                if str(candidate.get("candidate_id") or "") == selected_id:
                    if candidate_matches_canonical_selection(selection, candidate):
                        if candidate_px is not None:
                            cv2.circle(panel, candidate_px, max(5, int(round(3.0 * max(scale, 1.0)))), color, 2, cv2.LINE_AA)
                    else:
                        selected_candidate_stale = True
                if candidate_point is not None and candidate_px is not None and str(candidate.get("candidate_id") or "") != selected_id:
                    if behavior_type == "EXPLORE":
                        radius = max(3, int(round(1.3 * max(scale, 1.0))))
                        cv2.circle(panel, candidate_px, radius, UNSELECTED_EXPLORE_COLOR, -1, cv2.LINE_AA)
                        cv2.circle(panel, candidate_px, radius, UNSELECTED_EXPLORE_BORDER_COLOR, 1, cv2.LINE_AA)
                    else:
                        cv2.circle(panel, candidate_px, max(2, int(round(max(scale, 1.0) * 0.8))), color, -1, cv2.LINE_AA)

        # The live semantic selection (including an executor fallback) is the
        # only authority for the goal arrow. Candidate lists are merely a
        # diagnostic snapshot and must never substitute same-ID stale geometry.
        if (
            goal is not None
            and (goal_px := to_panel(goal)) is not None
        ):
            behavior = str(selection.get("behavior_type") or "NAVIGATE").upper()
            transformed_goal_yaw = goal[2] if math.isfinite(goal[2]) else goal_yaw
            _draw_goal_arrow(panel, goal_px, transformed_goal_yaw, max(9, int(9 * scale)), candidate_color(behavior))
        if selected_candidate_stale:
            cv2.rectangle(panel, (6, 101), (min(panel.shape[1] - 6, 302), 120), (255, 255, 255), -1)
            cv2.putText(
                panel,
                "CANDIDATE SNAPSHOT OUTDATED (LIVE GOAL SHOWN)",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (55, 55, 170),
                1,
                cv2.LINE_AA,
            )
        _draw_panel_title(panel, title, step_index)
        draw_map_snapshot_note(
            panel,
            snapshot_meta,
            display_stamp_sec=display_stamp_sec,
            selection_reason=snapshot_selection_reason,
            y=57 if kind != "costmap" else 42,
        )
        if "COSTMAP" in title.upper():
            _draw_costmap_legend(panel)
        elif draw_semantic_candidates:
            draw_terminal_status(panel, step)
            _draw_occupancy_candidate_legend(
                panel,
                str(selection.get("behavior_type") or ""),
            )
        return panel

    def _world_view(self, bounds: tuple[float, float, float, float], panel_size: tuple[int, int], *, margin: int, vertical_center: float = 0.5):
        width, height = panel_size
        min_x, min_y, max_x, max_y = bounds
        scale = min((width - 2 * margin) / max(max_x - min_x, 1e-6), (height - 2 * margin - 20) / max(max_y - min_y, 1e-6))
        center_x, center_y = (min_x + max_x) * 0.5, (min_y + max_y) * 0.5
        return scale, lambda x, y: (int(width * 0.5 + (x - center_x) * scale), int(height * vertical_center - (y - center_y) * scale))

    def _warp_grid(self, panel_size: tuple[int, int], grid: RawGrid | None, image: np.ndarray | None, to_px, background: tuple[int, int, int]) -> np.ndarray | None:
        if grid is None or image is None:
            return None
        source = np.float32([[0.0, grid.height - 1.0], [grid.width - 1.0, grid.height - 1.0], [0.0, 0.0]])
        destination = np.float32([
            to_px(*grid.world_from_cell(0.0, 0.0)),
            to_px(*grid.world_from_cell(grid.width - 1.0, 0.0)),
            to_px(*grid.world_from_cell(0.0, grid.height - 1.0)),
        ])
        transform = cv2.getAffineTransform(source, destination)
        return cv2.warpAffine(image, transform, panel_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=background)

    def _draw_world_overview_inset(
        self,
        panel: np.ndarray,
        occupancy: RawGrid | None,
        overview_bounds: tuple[float, float, float, float],
        detail_bounds: tuple[float, float, float, float],
        pose: tuple[float, float, float] | None,
    ) -> None:
        """Overlay a stable full-known-map inset and its current detailed view.

        The inset is derived only from the causal occupancy snapshot already
        used by the panel.  It is an offline diagnostic of explored coverage,
        not an oracle floorplan or a future map extent.
        """

        panel_h, panel_w = panel.shape[:2]
        inset_w = max(96, min(154, panel_w // 3))
        inset_h = max(72, min(112, panel_h // 3))
        inset = np.full((inset_h, inset_w, 3), 248, dtype=np.uint8)
        scale, to_px = self._world_view(
            overview_bounds,
            (inset_w, inset_h),
            margin=5,
            vertical_center=0.54,
        )
        if occupancy is not None:
            layer = self._warp_grid(
                (inset_w, inset_h),
                occupancy,
                _occupancy_base(occupancy),
                to_px,
                (248, 248, 248),
            )
            if layer is not None:
                inset = cv2.addWeighted(layer, 0.70, inset, 0.30, 0.0)
        cv2.rectangle(inset, (0, 0), (inset_w - 1, inset_h - 1), (55, 55, 55), 1)
        detail_min_x, detail_min_y, detail_max_x, detail_max_y = detail_bounds
        viewport = [
            to_px(detail_min_x, detail_min_y),
            to_px(detail_max_x, detail_min_y),
            to_px(detail_max_x, detail_max_y),
            to_px(detail_min_x, detail_max_y),
        ]
        cv2.polylines(
            inset,
            [np.asarray(viewport, dtype=np.int32)],
            True,
            (230, 30, 45),
            1,
            cv2.LINE_AA,
        )
        if pose is not None:
            _draw_robot_arrow(inset, to_px(pose[0], pose[1]), pose[2], 7)
        cv2.rectangle(inset, (3, 3), (inset_w - 4, 16), (255, 255, 255), -1)
        cv2.putText(
            inset,
            "MAP OVERVIEW",
            (6, 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
        x0, y0 = panel_w - inset_w - 7, panel_h - inset_h - 7
        backdrop = panel.copy()
        cv2.rectangle(
            backdrop,
            (x0 - 2, y0 - 2),
            (x0 + inset_w + 2, y0 + inset_h + 2),
            (255, 255, 255),
            -1,
        )
        cv2.addWeighted(backdrop, 0.70, panel, 0.30, 0.0, panel)
        panel[y0 : y0 + inset_h, x0 : x0 + inset_w] = inset

    def render_room_panel(
        self,
        occupancy: RawGrid | None,
        room: RawGrid | None,
        panel_size: tuple[int, int],
        step: dict,
        step_index: int,
        world_bounds: tuple[float, float, float, float] | None,
        *,
        view_scale: float = 1.0,
        snapshot_meta: dict | None = None,
        display_stamp_sec: float = 0.0,
        snapshot_selection_reason: str = "",
    ) -> np.ndarray:
        width, height = panel_size
        panel = np.full((height, width, 3), 246, dtype=np.uint8)
        reference = occupancy or room
        if reference is None:
            _draw_panel_title(panel, "ROOM SEGMENTS + INTERACTION", step_index)
            return panel
        if world_bounds is None:
            world_bounds = known_world_bounds(reference, 0.0) or (reference.origin_x, reference.origin_y, reference.origin_x + reference.width * reference.resolution, reference.origin_y + reference.height * reference.resolution)
        view_bounds = zoom_world_bounds(world_bounds, view_scale)
        scale, to_px = self._world_view(view_bounds, panel_size, margin=18, vertical_center=0.5)
        occ_layer = self._warp_grid(panel_size, occupancy, _occupancy_base(occupancy) if occupancy else None, to_px, (246, 246, 246))
        if occ_layer is not None:
            panel = cv2.addWeighted(occ_layer, 0.72, panel, 0.28, 0.0)
        room_layer = self._warp_grid(panel_size, room, _room_base(room) if room else None, to_px, (246, 246, 246))
        if room_layer is not None:
            panel = cv2.addWeighted(room_layer, 0.38, panel, 0.62, 0.0)
        graph = step.get("unified_graph") or {}
        selection = active_semantic_selection(step)
        target_ids = selection_target_ids(selection)
        observed = {str(value) for value in step.get("observed_instance_ids") or []}
        for node in _bounded_nodes(graph, observed, target_ids):
            if str(node.get("type") or "") not in {"portal", "container"}:
                continue
            center = _node_xy(node)
            size = list(node.get("aabb_size") or [])
            if center is None or len(size) < 2:
                continue
            center_px = to_px(*center)
            half_w, half_h = max(3, int(abs(float(size[0])) * scale * 0.5)), max(3, int(abs(float(size[1])) * scale * 0.5))
            is_target = node_matches_selection(node, target_ids)
            color = (235, 35, 210) if is_target else _node_color(node)
            cv2.rectangle(panel, (center_px[0] - half_w, center_px[1] - half_h), (center_px[0] + half_w, center_px[1] + half_h), color, 4 if is_target else 2, cv2.LINE_AA)
            cv2.putText(panel, f"{'INTERACT ' if is_target else ''}{_short_node_id(node)} {node.get('label', node.get('type', ''))}", (center_px[0] + 3, center_px[1] - half_h - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)
        pose = self._transform(step.get("pose"), self.transforms.odom_frame, self.transforms.map_frame, step_index)
        if pose is not None:
            _draw_robot_arrow(panel, to_px(pose[0], pose[1]), pose[2], 14)
        _draw_panel_title(panel, "ROOM SEGMENTS + INTERACTION", step_index)
        draw_map_snapshot_note(
            panel,
            snapshot_meta,
            display_stamp_sec=display_stamp_sec,
            selection_reason=snapshot_selection_reason,
            y=42,
        )
        return panel

    def render_semantic_xy(
        self,
        occupancy: RawGrid | None,
        panel_size: tuple[int, int],
        step: dict,
        step_index: int,
        world_bounds: tuple[float, float, float, float] | None,
        *,
        draw_object_labels: bool = True,
        draw_support_labels: bool = True,
        view_scale: float = 1.0,
        label_mode: str = "all",
        draw_overview_inset: bool = False,
        snapshot_meta: dict | None = None,
        display_stamp_sec: float = 0.0,
        snapshot_selection_reason: str = "",
    ) -> np.ndarray:
        width, height = panel_size
        panel = np.full((height, width, 3), 246, dtype=np.uint8)
        graph = step.get("unified_graph") or {}
        selection = active_semantic_selection(step)
        target_ids = selection_target_ids(selection)
        observed = {str(value) for value in step.get("observed_instance_ids") or []}
        nodes = _bounded_nodes(graph, observed, target_ids)
        positions = [position for node in nodes if (position := _node_xy(node)) is not None]
        pose = self._transform(step.get("pose"), self.transforms.odom_frame, self.transforms.map_frame, step_index)
        if pose is not None:
            positions.append((pose[0], pose[1]))
        if not positions:
            cv2.putText(panel, "WAITING FOR UNIFIED GRAPH", (40, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 90), 2, cv2.LINE_AA)
            _draw_panel_title(panel, "SEMANTIC XY", step_index)
            return panel
        if world_bounds is None:
            min_x, max_x = min(value[0] for value in positions), max(value[0] for value in positions)
            min_y, max_y = min(value[1] for value in positions), max(value[1] for value in positions)
            world_bounds = (min_x, min_y, max_x, max_y)
        view_bounds = zoom_world_bounds(world_bounds, view_scale)
        scale, to_px = self._world_view(view_bounds, panel_size, margin=38, vertical_center=0.53)
        occ_layer = self._warp_grid(panel_size, occupancy, _occupancy_base(occupancy) if occupancy else None, to_px, (246, 246, 246))
        if occ_layer is not None:
            panel = cv2.addWeighted(occ_layer, 0.35, panel, 0.65, 0.0)
        min_x, min_y, max_x, max_y = view_bounds
        for grid_x in range(math.floor(min_x), math.ceil(max_x) + 1):
            cv2.line(panel, to_px(grid_x, min_y), to_px(grid_x, max_y), (226, 226, 226), 1)
        for grid_y in range(math.floor(min_y), math.ceil(max_y) + 1):
            cv2.line(panel, to_px(min_x, grid_y), to_px(max_x, grid_y), (226, 226, 226), 1)
        lookup = {str(node.get("id") or ""): node for node in nodes}
        normalized_label_mode = str(label_mode or "all").casefold()
        if normalized_label_mode not in {"all", "interaction_target_only", "none"}:
            normalized_label_mode = "all"
        for edge in graph.get("edges") or []:
            if str(edge.get("relation") or "") not in {"connects", "contains", "supports"}:
                continue
            src, dst = _node_xy(lookup.get(str(edge.get("src_id") or ""), {})), _node_xy(lookup.get(str(edge.get("dst_id") or ""), {}))
            if src is None or dst is None:
                continue
            relation = str(edge.get("relation") or "")
            color = (220, 90, 45) if relation == "connects" else (170, 75, 210) if relation == "contains" else (55, 125, 220)
            cv2.line(panel, to_px(*src), to_px(*dst), color, 2, cv2.LINE_AA)
        for node in sorted(nodes, key=lambda item: str(item.get("type") or "") != "room"):
            center = _node_xy(node)
            if center is None:
                continue
            size = node.get("aabb_size") or [0.25, 0.25, 0.0]
            half_w, half_h = max(3, int(max(0.08, float(size[0]) * 0.5) * scale)), max(3, int(max(0.08, float(size[1]) * 0.5) * scale))
            pixel = to_px(*center)
            is_target = node_matches_selection(node, target_ids)
            color = (235, 35, 210) if is_target else _node_color(node)
            thickness = 4 if is_target else 2 if node.get("is_currently_visible") else 1
            if str(node.get("type") or "") == "room":
                overlay = panel.copy()
                cv2.rectangle(overlay, (pixel[0] - half_w, pixel[1] - half_h), (pixel[0] + half_w, pixel[1] + half_h), color, -1)
                cv2.addWeighted(overlay, 0.35, panel, 0.65, 0, panel)
                cv2.rectangle(panel, (pixel[0] - half_w, pixel[1] - half_h), (pixel[0] + half_w, pixel[1] + half_h), (125, 150, 175), 1)
            else:
                cv2.rectangle(panel, (pixel[0] - half_w, pixel[1] - half_h), (pixel[0] + half_w, pixel[1] + half_h), color, thickness)
            node_type = str(node.get("type") or "")
            draw_label = (
                node_type == "room"
                or (
                    normalized_label_mode == "interaction_target_only"
                    and is_target
                )
                or (
                    normalized_label_mode == "all"
                    and (node_type != "object" or draw_object_labels)
                    and (node_type != "support" or draw_support_labels)
                )
            )
            if draw_label:
                label = _node_label(node) if node_type == "room" else f"{'INTERACT ' if is_target else ''}{_short_node_id(node)} {_node_label(node)}"
                cv2.putText(panel, label[:30], (pixel[0] + 3, pixel[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, color if is_target else (35, 35, 35), 1, cv2.LINE_AA)
        if pose is not None:
            _draw_robot_arrow(panel, to_px(pose[0], pose[1]), pose[2], 14)
        if draw_overview_inset:
            self._draw_world_overview_inset(
                panel,
                occupancy,
                world_bounds,
                view_bounds,
                pose,
            )
        _draw_panel_title(panel, "SEMANTIC XY", step_index)
        draw_map_snapshot_note(
            panel,
            snapshot_meta,
            display_stamp_sec=display_stamp_sec,
            selection_reason=snapshot_selection_reason,
            y=42,
        )
        return panel

    def render_topology(self, panel_size: tuple[int, int], step: dict, step_index: int) -> np.ndarray:
        width, height = panel_size
        panel = np.full((height, width, 3), (220, 248, 255), dtype=np.uint8)
        graph = step.get("unified_graph") or {}
        selection = active_semantic_selection(step)
        target_ids = selection_target_ids(selection)
        observed = {str(value) for value in step.get("observed_instance_ids") or []}
        all_nodes = _bounded_nodes(graph, observed, target_ids)
        contains = [edge for edge in graph.get("edges") or [] if str(edge.get("relation") or "") == "contains"]
        contained = {str(edge.get("dst_id") or "") for edge in contains}
        nodes = [node for node in all_nodes if str(node.get("type") or "") in {"room", "portal", "container"} or str(node.get("id") or "") in contained]
        lookup = {str(node.get("id") or ""): node for node in nodes}
        edges = [dict(edge) for edge in graph.get("edges") or [] if topology_edge_visible(edge, lookup)]
        edge_keys = {(str(edge.get("src_id") or ""), str(edge.get("dst_id") or ""), str(edge.get("relation") or "")) for edge in edges}
        for node in nodes:
            node_id, node_type = str(node.get("id") or ""), str(node.get("type") or "")
            if node_type == "portal":
                for room_id in portal_room_node_ids(node, graph.get("edges") or []):
                    key = (node_id, room_id, "connects")
                    if room_id in lookup and key not in edge_keys:
                        edges.append({"src_id": node_id, "dst_id": room_id, "relation": "connects"})
                        edge_keys.add(key)
            if node_type == "container":
                room_id = node.get("room_id")
                parent = f"room_{int(room_id)}" if room_id is not None else str(node.get("parent_id") or "")
                key = (parent, node_id, "has_child")
                if parent in lookup and key not in edge_keys:
                    edges.append({"src_id": parent, "dst_id": node_id, "relation": "has_child"})
                    edge_keys.add(key)
        layout = topology_hierarchy_layout(nodes, edges, width, height)
        positions, boxes, row_y = layout["positions"], layout["boxes"], layout["row_y"]
        labels = {"room": "room level", "portal": "door level", "container": "container level", "object": "object level"}
        for node_type, y in row_y.items():
            cv2.line(panel, (layout["left_gutter"], y), (width - 5, y), (218, 221, 224), 1, cv2.LINE_AA)
            cv2.putText(panel, labels[node_type], (4, min(height - 4, y + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (120, 105, 90), 1, cv2.LINE_AA)
        for edge in edges:
            src_id, dst_id = str(edge.get("src_id") or ""), str(edge.get("dst_id") or "")
            if src_id not in positions or dst_id not in positions:
                continue
            style = topology_edge_style(str(edge.get("relation") or ""), str((lookup.get(src_id) or {}).get("type") or ""), str((lookup.get(dst_id) or {}).get("type") or ""))
            if style is not None:
                cv2.line(panel, positions[src_id], positions[dst_id], style["color"], int(style["thickness"]), cv2.LINE_AA)
        def dashed_line(start, end, color):
            dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
            length = max(1.0, math.hypot(dx, dy))
            cursor = 0.0
            while cursor < length:
                finish = min(length, cursor + 6)
                p1 = (int(start[0] + dx * cursor / length), int(start[1] + dy * cursor / length))
                p2 = (int(start[0] + dx * finish / length), int(start[1] + dy * finish / length))
                cv2.line(panel, p1, p2, color, 2, cv2.LINE_AA)
                cursor += 10
        for node in nodes:
            node_id = str(node.get("id") or "")
            box = boxes.get(node_id)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            style = topology_node_style(node)
            cv2.rectangle(panel, (x1, y1), (x2, y2), style["fill"], -1)
            if style["dashed"]:
                dashed_line((x1, y1), (x2, y1), style["border"]); dashed_line((x2, y1), (x2, y2), style["border"]); dashed_line((x2, y2), (x1, y2), style["border"]); dashed_line((x1, y2), (x1, y1), style["border"])
            else:
                cv2.rectangle(panel, (x1, y1), (x2, y2), style["border"], 2, cv2.LINE_AA)
            is_target = node_matches_selection(node, target_ids)
            if is_target:
                cv2.rectangle(panel, (max(1, x1 - 2), max(1, y1 - 2)), (min(width - 2, x2 + 2), min(height - 2, y2 + 2)), (235, 35, 210), 2, cv2.LINE_AA)
            node_type = str(node.get("type") or "object")
            state = str((node.get("interaction") or {}).get("state") or "unknown")
            if node_type == "room":
                room_id = node.get("room_id") if node.get("room_id") is not None else node_id.removeprefix("room_")
                cv2.putText(panel, f"Room {room_id}"[:22], (x1 + 4, y1 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (25, 25, 25), 1, cv2.LINE_AA)
                cv2.putText(panel, _node_label(node)[:22], (x1 + 4, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (25, 25, 25), 1, cv2.LINE_AA)
            else:
                label = _node_label(node) if node_type == "object" else f"{_short_node_id(node)} {_node_label(node)} [{state}]"
                text_width, text_height = cv2.getTextSize(label[:26], cv2.FONT_HERSHEY_SIMPLEX, 0.25, 1)[0]
                cv2.putText(panel, label[:26], (x1 + max(3, (x2 - x1 - text_width) // 2), (y1 + y2 + text_height) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (25, 25, 25), 1, cv2.LINE_AA)
        dashed_line((4, 29), (13, 29), (35, 35, 210)); dashed_line((13, 29), (13, 37), (35, 35, 210)); dashed_line((13, 37), (4, 37), (35, 35, 210)); dashed_line((4, 37), (4, 29), (35, 35, 210))
        cv2.putText(panel, "closed", (16, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.23, (65, 65, 65), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (57, 29), (66, 37), (45, 175, 70), 2)
        cv2.putText(panel, "open", (69, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.23, (65, 65, 65), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (106, 29), (115, 37), (195, 175, 35), 2)
        cv2.putText(panel, "static open", (118, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.23, (65, 65, 65), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (193, 29), (202, 37), (175, 45, 185), 2)
        cv2.putText(panel, "blocked", (205, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.23, (65, 65, 65), 1, cv2.LINE_AA)
        execution = str((step.get("semantic_execution_state") or {}).get("state") or "IDLE")
        behavior = str(selection.get("behavior_type") or "-")
        feedback = str((step.get("semantic_behavior_feedback") or {}).get("status") or "-")
        cv2.putText(panel, f"{execution}/{behavior}  feedback={feedback}"[:72], (5, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.24, (55, 55, 55), 1, cv2.LINE_AA)
        revision = int(graph.get("graph_revision", 0) or 0)
        _draw_panel_title(panel, f"SEMANTIC INTERACTION GRAPH r{revision}", step_index)
        return panel


def camera_title(step: dict, step_index: int) -> str:
    pose = step.get("pose") or []
    raw_selection = step.get("semantic_selection") or {}
    selection = active_semantic_selection(step)
    goal = list(
        []
        if raw_selection.get("active") is False
        else selection.get("goal_xyyaw") or step.get("active_goal") or []
    )
    distance = float(step.get("distance_m") or 0.0)
    if len(pose) >= 2 and len(goal) >= 2:
        goal_distance = f"{math.hypot(float(pose[0]) - float(goal[0]), float(pose[1]) - float(goal[1])):.2f}m"
    else:
        goal_distance = "-"
    return f"STEP={_step4(step_index)}  dist={distance:.2f}m  dist_to_goal={goal_distance}"


def draw_camera_title(frame: np.ndarray, step: dict, step_index: int) -> None:
    text = camera_title(step, step_index)
    font_scale = 0.46 if frame.shape[1] < 700 else 0.58
    thickness = 1 if frame.shape[1] < 700 else 2
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x = max(6, frame.shape[1] - text_size[0] - 8)
    y = max(text_size[1] + 5, 22)
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 245, 245), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (25, 25, 25), thickness, cv2.LINE_AA)
