#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import struct
import sys
import threading
import time
import zlib
from pathlib import Path


def _patch_roslogging_findcaller_for_py311() -> None:
    if sys.version_info < (3, 11):
        return
    try:
        import rosgraph.roslogging
    except Exception:
        return

    def find_caller(self, stack_info=False, stacklevel=1):  # noqa: ANN001
        frame = logging.currentframe()
        if frame is not None:
            frame = frame.f_back
        while frame and stacklevel > 1:
            frame = frame.f_back
            stacklevel -= 1
        if frame is None:
            return "(unknown file)", 0, "(unknown function)", None
        code = frame.f_code
        return code.co_filename, frame.f_lineno, code.co_name, None

    rosgraph.roslogging.RospyLogger.findCaller = find_caller


_patch_roslogging_findcaller_for_py311()

import rospy
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PointStamped, PoseStamped, Twist, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from rosgraph_msgs.msg import Log
from sensor_msgs.msg import Image
from std_msgs.msg import String


STATUS_NAMES = {
    0: "PENDING",
    1: "ACTIVE",
    2: "PREEMPTED",
    3: "SUCCEEDED",
    4: "ABORTED",
    5: "REJECTED",
    6: "PREEMPTING",
    7: "RECALLING",
    8: "RECALLED",
    9: "LOST",
}


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _pose_xy_yaw(msg: Odometry) -> tuple[float, float, float]:
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation
    return float(position.x), float(position.y), _yaw_from_quaternion(orientation)


def _grid_origin_yaw(grid: OccupancyGrid) -> float:
    return _yaw_from_quaternion(grid.info.origin.orientation)


def _world_to_cell(grid: OccupancyGrid, x: float, y: float) -> tuple[int, int] | None:
    resolution = float(grid.info.resolution)
    if resolution <= 0.0:
        return None
    origin = grid.info.origin.position
    yaw = _grid_origin_yaw(grid)
    dx = x - float(origin.x)
    dy = y - float(origin.y)
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    local_x = cos_yaw * dx - sin_yaw * dy
    local_y = sin_yaw * dx + cos_yaw * dy
    mx = int(math.floor(local_x / resolution))
    my = int(math.floor(local_y / resolution))
    if mx < 0 or my < 0 or mx >= int(grid.info.width) or my >= int(grid.info.height):
        return None
    return mx, my


def _grid_value(grid: OccupancyGrid, cell: tuple[int, int] | None) -> int | None:
    if cell is None:
        return None
    x, y = cell
    return int(grid.data[y * int(grid.info.width) + x])


def _is_free(value: int | None) -> bool:
    return value is not None and 0 <= value <= 20


def _is_unknown(value: int | None) -> bool:
    return value is not None and value < 0


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_png(path: Path, width: int, height: int, rgb: bytearray) -> None:
    rows = []
    stride = width * 3
    for row in range(height):
        start = row * stride
        rows.append(b"\x00" + bytes(rgb[start : start + stride]))
    payload = b"".join(rows)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n"
    data += _chunk(b"IHDR", header)
    data += _chunk(b"IDAT", zlib.compress(payload, 6))
    data += _chunk(b"IEND", b"")
    path.write_bytes(data)


def _read_png(path: Path) -> tuple[int, int, bytearray] | None:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    offset = 8
    width = 0
    height = 0
    idat = bytearray()
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", payload)
            if bit_depth != 8 or color_type != 2 or compression != 0 or filter_method != 0 or interlace != 0:
                return None
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            break
    if width <= 0 or height <= 0:
        return None
    raw = zlib.decompress(bytes(idat))
    stride = width * 3
    expected = height * (stride + 1)
    if len(raw) != expected:
        return None
    rgb = bytearray(width * height * 3)
    for y in range(height):
        row_start = y * (stride + 1)
        if raw[row_start] != 0:
            return None
        rgb[y * stride : (y + 1) * stride] = raw[row_start + 1 : row_start + 1 + stride]
    return width, height, rgb


def _image_msg_to_rgb(msg: Image) -> tuple[int, int, bytearray] | None:
    width = int(msg.width)
    height = int(msg.height)
    if width <= 0 or height <= 0:
        return None
    encoding = (msg.encoding or "").lower()
    step = int(msg.step)
    data = bytes(msg.data)
    rgb = bytearray(width * height * 3)
    if encoding in ("rgb8", "bgr8"):
        channels = 3
        min_step = width * channels
        if step < min_step or len(data) < step * height:
            return None
        for y in range(height):
            src_row = y * step
            dst_row = y * width * 3
            for x in range(width):
                src = src_row + x * channels
                dst = dst_row + x * 3
                if encoding == "rgb8":
                    rgb[dst : dst + 3] = data[src : src + 3]
                else:
                    rgb[dst : dst + 3] = bytes((data[src + 2], data[src + 1], data[src]))
        return width, height, rgb
    if encoding in ("rgba8", "bgra8"):
        channels = 4
        min_step = width * channels
        if step < min_step or len(data) < step * height:
            return None
        for y in range(height):
            src_row = y * step
            dst_row = y * width * 3
            for x in range(width):
                src = src_row + x * channels
                dst = dst_row + x * 3
                if encoding == "rgba8":
                    rgb[dst : dst + 3] = data[src : src + 3]
                else:
                    rgb[dst : dst + 3] = bytes((data[src + 2], data[src + 1], data[src]))
        return width, height, rgb
    if encoding in ("mono8", "8uc1"):
        if step < width or len(data) < step * height:
            return None
        for y in range(height):
            src_row = y * step
            dst_row = y * width * 3
            for x in range(width):
                value = data[src_row + x]
                dst = dst_row + x * 3
                rgb[dst : dst + 3] = bytes((value, value, value))
        return width, height, rgb
    return None


def _crop_rgb_to_content(
    rgb: bytearray,
    width: int,
    height: int,
    background: tuple[int, int, int] = (178, 178, 178),
    margin_px: int = 80,
) -> tuple[int, int, bytearray] | None:
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    bg = bytes(background)
    for y in range(height):
        row_start = y * width * 3
        for x in range(width):
            index = row_start + x * 3
            if rgb[index : index + 3] != bg:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    if max_x < min_x or max_y < min_y:
        return None
    min_x = max(0, min_x - margin_px)
    min_y = max(0, min_y - margin_px)
    max_x = min(width - 1, max_x + margin_px)
    max_y = min(height - 1, max_y + margin_px)
    crop_width = max_x - min_x + 1
    crop_height = max_y - min_y + 1
    cropped = bytearray(crop_width * crop_height * 3)
    for y in range(crop_height):
        src_start = ((min_y + y) * width + min_x) * 3
        dst_start = y * crop_width * 3
        cropped[dst_start : dst_start + crop_width * 3] = rgb[src_start : src_start + crop_width * 3]
    return crop_width, crop_height, cropped


def _resize_rgb_nearest(
    rgb: bytearray,
    width: int,
    height: int,
    target_width: int | None = None,
    target_height: int | None = None,
) -> tuple[int, int, bytearray]:
    if width <= 0 or height <= 0:
        return 0, 0, bytearray()
    if target_width is None and target_height is None:
        return width, height, rgb
    if target_width is None:
        scale = float(target_height) / float(height)
        target_width = max(1, int(round(width * scale)))
    if target_height is None:
        scale = float(target_width) / float(width)
        target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(target_width))
    target_height = max(1, int(target_height))
    if target_width == width and target_height == height:
        return width, height, rgb
    resized = bytearray(target_width * target_height * 3)
    for y in range(target_height):
        src_y = min(height - 1, int(y * height / target_height))
        for x in range(target_width):
            src_x = min(width - 1, int(x * width / target_width))
            src = (src_y * width + src_x) * 3
            dst = (y * target_width + x) * 3
            resized[dst : dst + 3] = rgb[src : src + 3]
    return target_width, target_height, resized


def _scale_rgb_nearest(rgb: bytearray, width: int, height: int, scale: int) -> tuple[int, int, bytearray]:
    scale = max(1, int(scale))
    if scale == 1:
        return width, height, rgb
    scaled_width = width * scale
    scaled_height = height * scale
    scaled = bytearray(scaled_width * scaled_height * 3)
    for y in range(height):
        for repeat_y in range(scale):
            dst_row = (y * scale + repeat_y) * scaled_width * 3
            for x in range(width):
                pixel = rgb[(y * width + x) * 3 : (y * width + x + 1) * 3]
                for repeat_x in range(scale):
                    dst_index = dst_row + (x * scale + repeat_x) * 3
                    scaled[dst_index : dst_index + 3] = pixel
    return scaled_width, scaled_height, scaled


FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "#": ["01010", "11111", "01010", "01010", "11111", "01010", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "11100"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00001", "00001", "00001", "00001", "10001", "10001", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def _draw_text(
    rgb: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int] = (20, 20, 20),
    scale: int = 2,
) -> None:
    cursor_x = x
    scale = max(1, int(scale))
    for char in text.upper():
        glyph = FONT_5X7.get(char, FONT_5X7[" "])
        for gy, row in enumerate(glyph):
            for gx, value in enumerate(row):
                if value != "1":
                    continue
                for sy in range(scale):
                    py = y + gy * scale + sy
                    if py < 0 or py >= height:
                        continue
                    for sx in range(scale):
                        px = cursor_x + gx * scale + sx
                        if px < 0 or px >= width:
                            continue
                        index = (py * width + px) * 3
                        rgb[index : index + 3] = bytes(color)
        cursor_x += 6 * scale


def _paste_rgb(
    canvas: bytearray,
    canvas_width: int,
    canvas_height: int,
    image: bytearray,
    image_width: int,
    image_height: int,
    x0: int,
    y0: int,
) -> None:
    for y in range(image_height):
        dst_y = y0 + y
        if dst_y < 0 or dst_y >= canvas_height:
            continue
        src_start = y * image_width * 3
        dst_start = (dst_y * canvas_width + x0) * 3
        if x0 < 0:
            src_start += -x0 * 3
            dst_start = dst_y * canvas_width * 3
            copy_width = min(image_width + x0, canvas_width)
        else:
            copy_width = min(image_width, canvas_width - x0)
        if copy_width <= 0:
            continue
        canvas[dst_start : dst_start + copy_width * 3] = image[src_start : src_start + copy_width * 3]


def _make_side_by_side_panel(
    left: tuple[int, int, bytearray],
    right: tuple[int, int, bytearray] | None,
    title: str,
    image_height: int,
    gap_px: int,
    title_height_px: int,
) -> tuple[int, int, bytearray]:
    left_w, left_h, left_rgb = left
    left_w, left_h, left_rgb = _resize_rgb_nearest(left_rgb, left_w, left_h, target_height=image_height)
    if right is None:
        right_w, right_h, right_rgb = left_w, image_height, bytearray([235, 235, 235] * left_w * image_height)
        _draw_text(right_rgb, right_w, right_h, 16, 16, "NO IMAGE", (120, 120, 120), scale=3)
    else:
        right_w, right_h, right_rgb = right
        right_w, right_h, right_rgb = _resize_rgb_nearest(right_rgb, right_w, right_h, target_height=image_height)

    width = left_w + gap_px + right_w
    height = title_height_px + image_height
    canvas = bytearray([255, 255, 255] * width * height)
    _draw_text(canvas, width, height, 12, 8, title, (20, 20, 20), scale=2)
    _paste_rgb(canvas, width, height, left_rgb, left_w, left_h, 0, title_height_px)
    _paste_rgb(canvas, width, height, right_rgb, right_w, right_h, left_w + gap_px, title_height_px)
    return width, height, canvas


def _make_contact_sheet(
    panels: list[tuple[int, int, bytearray]],
    columns: int,
    gap_px: int,
    background: tuple[int, int, int] = (245, 245, 242),
) -> tuple[int, int, bytearray] | None:
    if not panels:
        return None
    columns = max(1, int(columns))
    rows = int(math.ceil(len(panels) / columns))
    cell_width = max(width for width, _, _ in panels)
    cell_height = max(height for _, height, _ in panels)
    width = columns * cell_width + (columns - 1) * gap_px
    height = rows * cell_height + (rows - 1) * gap_px
    canvas = bytearray(list(background) * width * height)
    for index, (panel_width, panel_height, panel_rgb) in enumerate(panels):
        row = index // columns
        col = index % columns
        x = col * (cell_width + gap_px)
        y = row * (cell_height + gap_px)
        _paste_rgb(canvas, width, height, panel_rgb, panel_width, panel_height, x, y)
    return width, height, canvas


def _bresenham(x0: int, y0: int, x1: int, y1: int):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        err2 = 2 * err
        if err2 >= dy:
            err += dy
            x0 += sx
        if err2 <= dx:
            err += dx
            y0 += sy


class ExploreDebugRecorder:
    def __init__(self, output_dir: Path, args: argparse.Namespace):
        self.output_dir = output_dir
        self.overlay_dir = output_dir / "subgoal_overlays"
        self.first_person_dir = output_dir / "first_person"
        self.panel_dir = output_dir / "subgoal_panels"
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        self.first_person_dir.mkdir(parents=True, exist_ok=True)
        self.panel_dir.mkdir(parents=True, exist_ok=True)

        self.args = args
        self.lock = threading.RLock()
        self.shutting_down = False
        self.start_wall_time = time.time()
        self.latest_grid: OccupancyGrid | None = None
        self.latest_image: tuple[float, int, int, bytearray] | None = None
        self.latest_pose: tuple[float, float, float] | None = None
        self.latest_global_plan: dict | None = None
        self.latest_local_plan: dict | None = None
        self.last_odom_xy: tuple[float, float] | None = None
        self.last_recorded_odom_time = 0.0
        self.distance_m = 0.0
        self.trajectory: list[tuple[float, float, float, float]] = []
        self.goal_count = 0
        self.current_subgoal_count = 0
        self.last_status_key = ""
        self.seen_status_keys: set[str] = set()
        self.last_explore_status_time = 0.0
        self.last_explore_active_goal = None
        self.last_cmd_vel_record_time: dict[str, float] = {}
        self.cmd_vel_counts: dict[str, int] = {}
        self.cmd_vel_nonzero_counts: dict[str, int] = {}
        self.cmd_vel_max_speed: dict[str, float] = {}
        self.status_counts: dict[str, int] = {}
        self.plan_message_counts = {"global": 0, "local": 0}
        self.subgoal_records: list[dict] = []

        self.events_file = (output_dir / "events.jsonl").open("a", buffering=1)
        self.trajectory_file = (output_dir / "trajectory.csv").open("a", newline="", buffering=1)
        self.subgoals_file = (output_dir / "subgoals.csv").open("a", newline="", buffering=1)
        self.status_file = (output_dir / "move_base_status.csv").open("a", newline="", buffering=1)
        self.cmd_vel_file = (output_dir / "cmd_vel.csv").open("a", newline="", buffering=1)
        self.plan_file = (output_dir / "move_base_plans.csv").open("a", newline="", buffering=1)
        self.move_base_log_file = (output_dir / "move_base_rosout.log").open("a", buffering=1)

        self.trajectory_writer = csv.DictWriter(
            self.trajectory_file,
            fieldnames=["elapsed_sec", "stamp", "x", "y", "yaw", "step_distance_m", "total_distance_m"],
        )
        self.subgoals_writer = csv.DictWriter(
            self.subgoals_file,
            fieldnames=[
                "index",
                "elapsed_sec",
                "stamp",
                "x",
                "y",
                "robot_x",
                "robot_y",
                "robot_distance_m",
                "goal_cell_x",
                "goal_cell_y",
                "goal_cell_value",
                "goal_is_free",
                "unknown_cells_near_goal",
                "nearest_unknown_m",
                "frontier_like",
                "overlay",
                "overlay_crop",
                "first_person",
                "first_person_stamp",
                "panel",
                "global_plan_points",
                "global_plan_stamp",
                "local_plan_points",
                "local_plan_stamp",
            ],
        )
        self.status_writer = csv.DictWriter(
            self.status_file,
            fieldnames=["elapsed_sec", "stamp", "goal_id", "status", "status_name", "text"],
        )
        self.cmd_vel_writer = csv.DictWriter(
            self.cmd_vel_file,
            fieldnames=["elapsed_sec", "stamp", "topic", "linear_x", "linear_y", "angular_z", "speed"],
        )
        self.plan_writer = csv.DictWriter(
            self.plan_file,
            fieldnames=["elapsed_sec", "topic", "plan_type", "message_index", "stamp", "frame_id", "pose_index", "pose_count", "x", "y", "yaw"],
        )
        self.trajectory_writer.writeheader()
        self.subgoals_writer.writeheader()
        self.status_writer.writeheader()
        self.cmd_vel_writer.writeheader()
        self.plan_writer.writeheader()

        rospy.Subscriber(args.occupancy_grid_topic, OccupancyGrid, self.occupancy_callback, queue_size=1)
        rospy.Subscriber(args.image_topic, Image, self.image_callback, queue_size=1)
        rospy.Subscriber(args.odom_topic, Odometry, self.odom_callback, queue_size=50)
        rospy.Subscriber(args.goal_topic, PoseStamped, self.goal_callback, queue_size=20)
        rospy.Subscriber(args.current_subgoal_topic, PointStamped, self.current_subgoal_callback, queue_size=20)
        rospy.Subscriber(args.move_base_status_topic, GoalStatusArray, self.move_base_status_callback, queue_size=20)
        rospy.Subscriber(args.explore_status_topic, String, self.explore_status_callback, queue_size=20)
        rospy.Subscriber(args.cmd_vel_topic, Twist, self.cmd_vel_callback, callback_args=args.cmd_vel_topic, queue_size=50)
        rospy.Subscriber(args.global_plan_topic, NavPath, self.plan_callback, callback_args=("global", args.global_plan_topic), queue_size=20)
        rospy.Subscriber(args.local_plan_topic, NavPath, self.plan_callback, callback_args=("local", args.local_plan_topic), queue_size=20)
        rospy.Subscriber(args.rosout_topic, Log, self.rosout_callback, queue_size=200)
        rospy.Subscriber(
            args.cmd_vel_stamped_topic,
            TwistStamped,
            self.cmd_vel_stamped_callback,
            callback_args=args.cmd_vel_stamped_topic,
            queue_size=50,
        )
        rospy.on_shutdown(self.shutdown)

    def occupancy_callback(self, msg: OccupancyGrid) -> None:
        with self.lock:
            self.latest_grid = msg

    def image_callback(self, msg: Image) -> None:
        if self.shutting_down:
            return
        converted = _image_msg_to_rgb(msg)
        if converted is None:
            return
        width, height, rgb = converted
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
        with self.lock:
            self.latest_image = (stamp, width, height, rgb)

    def odom_callback(self, msg: Odometry) -> None:
        if self.shutting_down:
            return
        x, y, yaw = _pose_xy_yaw(msg)
        now = time.time()
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
        with self.lock:
            step = 0.0
            if self.last_odom_xy is not None:
                step = math.hypot(x - self.last_odom_xy[0], y - self.last_odom_xy[1])
                if step <= self.args.max_odom_jump_m:
                    self.distance_m += step
                else:
                    self._write_event(
                        "odom_jump_ignored",
                        {
                            "from": list(self.last_odom_xy),
                            "to": [x, y],
                            "step_distance_m": step,
                            "max_odom_jump_m": self.args.max_odom_jump_m,
                        },
                    )
                    step = 0.0
            self.latest_pose = (x, y, yaw)
            self.last_odom_xy = (x, y)
            should_record = (
                not self.trajectory
                or now - self.last_recorded_odom_time >= self.args.trajectory_period_sec
                or step >= self.args.trajectory_min_step_m
            )
            if should_record:
                elapsed = now - self.start_wall_time
                self.trajectory.append((elapsed, x, y, yaw))
                self.last_recorded_odom_time = now
                self.trajectory_writer.writerow(
                    {
                        "elapsed_sec": f"{elapsed:.3f}",
                        "stamp": f"{stamp:.6f}",
                        "x": f"{x:.6f}",
                        "y": f"{y:.6f}",
                        "yaw": f"{yaw:.6f}",
                        "step_distance_m": f"{step:.6f}",
                        "total_distance_m": f"{self.distance_m:.6f}",
                    }
                )

    def goal_callback(self, msg: PoseStamped) -> None:
        if self.shutting_down:
            return
        goal_xy = (float(msg.pose.position.x), float(msg.pose.position.y))
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
        with self.lock:
            self.goal_count += 1
            grid = self.latest_grid
            pose = self.latest_pose
            image_snapshot = self.latest_image
            trajectory = list(self.trajectory)
            global_plan = self.latest_global_plan
            local_plan = self.latest_local_plan
            analysis = self._analyze_goal(grid, pose, goal_xy)
            overlay_path = ""
            overlay_crop_path = ""
            first_person_path = ""
            first_person_stamp = 0.0
            panel_path = ""
            if grid is not None:
                overlay_path = str(self.overlay_dir / f"subgoal_{self.goal_count:04d}.png")
                overlay_crop_path = self._render_overlay(
                    Path(overlay_path),
                    grid,
                    pose,
                    goal_xy,
                    trajectory,
                    global_plan=global_plan,
                    local_plan=local_plan,
                )
            elapsed = time.time() - self.start_wall_time
            if image_snapshot is not None:
                first_person_stamp, image_width, image_height, image_rgb = image_snapshot
                first_person_path = str(self.first_person_dir / f"subgoal_{self.goal_count:04d}_first_person.png")
                _write_png(Path(first_person_path), image_width, image_height, image_rgb)
            global_plan_stamp = 0.0 if global_plan is None else float(global_plan.get("stamp", 0.0))
            local_plan_stamp = 0.0 if local_plan is None else float(local_plan.get("stamp", 0.0))
            global_plan_points = 0 if global_plan is None else len(global_plan.get("poses", []))
            local_plan_points = 0 if local_plan is None else len(local_plan.get("poses", []))
            panel_path = self._render_subgoal_panel(
                self.goal_count,
                elapsed,
                stamp,
                first_person_stamp,
                overlay_crop_path or overlay_path,
                first_person_path,
            )
            record = {
                "index": self.goal_count,
                "elapsed_sec": elapsed,
                "stamp": stamp,
                "goal": list(goal_xy),
                "robot_pose": list(pose) if pose is not None else None,
                "analysis": analysis,
                "overlay": overlay_path,
                "overlay_crop": overlay_crop_path,
                "first_person": first_person_path,
                "first_person_stamp": first_person_stamp,
                "panel": panel_path,
                "global_plan_points": global_plan_points,
                "global_plan_stamp": global_plan_stamp,
                "local_plan_points": local_plan_points,
                "local_plan_stamp": local_plan_stamp,
            }
            self.subgoal_records.append(record)
            self.subgoals_writer.writerow(
                {
                    "index": self.goal_count,
                    "elapsed_sec": f"{elapsed:.3f}",
                    "stamp": f"{stamp:.6f}",
                    "x": f"{goal_xy[0]:.6f}",
                    "y": f"{goal_xy[1]:.6f}",
                    "robot_x": "" if pose is None else f"{pose[0]:.6f}",
                    "robot_y": "" if pose is None else f"{pose[1]:.6f}",
                    "robot_distance_m": "" if analysis["robot_distance_m"] is None else f"{analysis['robot_distance_m']:.6f}",
                    "goal_cell_x": "" if analysis["goal_cell"] is None else analysis["goal_cell"][0],
                    "goal_cell_y": "" if analysis["goal_cell"] is None else analysis["goal_cell"][1],
                    "goal_cell_value": "" if analysis["goal_cell_value"] is None else analysis["goal_cell_value"],
                    "goal_is_free": analysis["goal_is_free"],
                    "unknown_cells_near_goal": analysis["unknown_cells_near_goal"],
                    "nearest_unknown_m": "" if analysis["nearest_unknown_m"] is None else f"{analysis['nearest_unknown_m']:.6f}",
                    "frontier_like": analysis["frontier_like"],
                    "overlay": overlay_path,
                    "overlay_crop": overlay_crop_path,
                    "first_person": first_person_path,
                    "first_person_stamp": "" if first_person_stamp <= 0.0 else f"{first_person_stamp:.6f}",
                    "panel": panel_path,
                    "global_plan_points": global_plan_points,
                    "global_plan_stamp": "" if global_plan_stamp <= 0.0 else f"{global_plan_stamp:.6f}",
                    "local_plan_points": local_plan_points,
                    "local_plan_stamp": "" if local_plan_stamp <= 0.0 else f"{local_plan_stamp:.6f}",
                }
            )
            self._write_event("subgoal_published", record)

    def current_subgoal_callback(self, msg: PointStamped) -> None:
        if self.shutting_down:
            return
        with self.lock:
            self.current_subgoal_count += 1
            self._write_event(
                "current_subgoal",
                {
                    "index": self.current_subgoal_count,
                    "stamp": msg.header.stamp.to_sec() if msg.header.stamp else 0.0,
                    "point": [float(msg.point.x), float(msg.point.y), float(msg.point.z)],
                },
            )

    def move_base_status_callback(self, msg: GoalStatusArray) -> None:
        if self.shutting_down:
            return
        if not msg.status_list:
            return
        with self.lock:
            for status in msg.status_list:
                code = int(status.status)
                status_name = STATUS_NAMES.get(code, str(code))
                goal_id = getattr(status.goal_id, "id", "")
                text = getattr(status, "text", "")
                key = f"{goal_id}:{code}:{text}"
                if key in self.seen_status_keys:
                    continue
                self.seen_status_keys.add(key)
                self.last_status_key = key
                self.status_counts[status_name] = self.status_counts.get(status_name, 0) + 1
                elapsed = time.time() - self.start_wall_time
                stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
                row = {
                    "elapsed_sec": f"{elapsed:.3f}",
                    "stamp": f"{stamp:.6f}",
                    "goal_id": goal_id,
                    "status": code,
                    "status_name": status_name,
                    "text": text,
                }
                self.status_writer.writerow(row)
                self._write_event(
                    "move_base_status",
                    {
                        "elapsed_sec": elapsed,
                        "stamp": stamp,
                        "goal_id": goal_id,
                        "status": code,
                        "status_name": status_name,
                        "text": text,
                    },
                )

    def cmd_vel_callback(self, msg: Twist, topic: str) -> None:
        self._record_cmd_vel(topic, 0.0, float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))

    def cmd_vel_stamped_callback(self, msg: TwistStamped, topic: str) -> None:
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
        self._record_cmd_vel(
            topic,
            stamp,
            float(msg.twist.linear.x),
            float(msg.twist.linear.y),
            float(msg.twist.angular.z),
        )

    def _record_cmd_vel(self, topic: str, stamp: float, linear_x: float, linear_y: float, angular_z: float) -> None:
        if self.shutting_down:
            return
        now = time.time()
        speed = math.hypot(linear_x, linear_y)
        is_nonzero = speed > self.args.cmd_vel_nonzero_threshold or abs(angular_z) > self.args.cmd_vel_nonzero_threshold
        with self.lock:
            self.cmd_vel_counts[topic] = self.cmd_vel_counts.get(topic, 0) + 1
            if is_nonzero:
                self.cmd_vel_nonzero_counts[topic] = self.cmd_vel_nonzero_counts.get(topic, 0) + 1
            self.cmd_vel_max_speed[topic] = max(self.cmd_vel_max_speed.get(topic, 0.0), speed)
            last_time = self.last_cmd_vel_record_time.get(topic, 0.0)
            if not is_nonzero and now - last_time < self.args.cmd_vel_record_period_sec:
                return
            if is_nonzero or now - last_time >= self.args.cmd_vel_record_period_sec:
                self.last_cmd_vel_record_time[topic] = now
                elapsed = now - self.start_wall_time
                self.cmd_vel_writer.writerow(
                    {
                        "elapsed_sec": f"{elapsed:.3f}",
                        "stamp": f"{stamp:.6f}",
                        "topic": topic,
                        "linear_x": f"{linear_x:.6f}",
                        "linear_y": f"{linear_y:.6f}",
                        "angular_z": f"{angular_z:.6f}",
                        "speed": f"{speed:.6f}",
                    }
                )

    def plan_callback(self, msg: NavPath, callback_args: tuple[str, str]) -> None:
        if self.shutting_down:
            return
        plan_type, topic = callback_args
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
        poses = []
        elapsed = time.time() - self.start_wall_time
        message_index = 0
        with self.lock:
            self.plan_message_counts[plan_type] = self.plan_message_counts.get(plan_type, 0) + 1
            message_index = self.plan_message_counts[plan_type]
        for pose_index, pose in enumerate(msg.poses):
            x = float(pose.pose.position.x)
            y = float(pose.pose.position.y)
            yaw = _yaw_from_quaternion(pose.pose.orientation)
            poses.append((x, y, yaw))
            with self.lock:
                self.plan_writer.writerow(
                    {
                        "elapsed_sec": f"{elapsed:.3f}",
                        "topic": topic,
                        "plan_type": plan_type,
                        "message_index": message_index,
                        "stamp": f"{stamp:.6f}",
                        "frame_id": msg.header.frame_id or "",
                        "pose_index": pose_index,
                        "pose_count": len(msg.poses),
                        "x": f"{x:.6f}",
                        "y": f"{y:.6f}",
                        "yaw": f"{yaw:.6f}",
                    }
                )
        snapshot = {
            "topic": topic,
            "stamp": stamp,
            "frame_id": msg.header.frame_id or "",
            "poses": poses,
            "message_index": message_index,
        }
        with self.lock:
            if plan_type == "global":
                self.latest_global_plan = snapshot
            else:
                self.latest_local_plan = snapshot
            self._refresh_latest_subgoal_overlay_locked()
            self._write_event(
                "plan_update",
                {
                    "elapsed_sec": elapsed,
                    "plan_type": plan_type,
                    "topic": topic,
                    "stamp": stamp,
                    "pose_count": len(poses),
                    "message_index": message_index,
                },
            )

    def _refresh_latest_subgoal_overlay_locked(self) -> None:
        if not self.subgoal_records or self.latest_grid is None:
            return
        record = self.subgoal_records[-1]
        if not isinstance(record, dict):
            return
        global_plan = self.latest_global_plan
        local_plan = self.latest_local_plan
        have_global = bool(global_plan and global_plan.get("poses"))
        have_local = bool(local_plan and local_plan.get("poses"))
        if not have_global and not have_local:
            return
        current_global = int(record.get("global_plan_points") or 0)
        current_local = int(record.get("local_plan_points") or 0)
        if current_global > 0 and current_local > 0:
            return
        goal = record.get("goal")
        if not goal or len(goal) < 2:
            return
        robot_pose = record.get("robot_pose")
        pose = None
        if isinstance(robot_pose, list) and len(robot_pose) >= 3:
            pose = (float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2]))
        overlay_path = record.get("overlay")
        if overlay_path:
            overlay_crop_path = self._render_overlay(
                Path(overlay_path),
                self.latest_grid,
                pose,
                (float(goal[0]), float(goal[1])),
                list(self.trajectory),
                global_plan=global_plan,
                local_plan=local_plan,
            )
            record["overlay_crop"] = overlay_crop_path
        global_plan_stamp = 0.0 if global_plan is None else float(global_plan.get("stamp", 0.0))
        local_plan_stamp = 0.0 if local_plan is None else float(local_plan.get("stamp", 0.0))
        global_plan_points = 0 if global_plan is None else len(global_plan.get("poses", []))
        local_plan_points = 0 if local_plan is None else len(local_plan.get("poses", []))
        record["global_plan_points"] = global_plan_points
        record["global_plan_stamp"] = global_plan_stamp
        record["local_plan_points"] = local_plan_points
        record["local_plan_stamp"] = local_plan_stamp
        panel_path = self._render_subgoal_panel(
            int(record["index"]),
            float(record["elapsed_sec"]),
            float(record.get("stamp", 0.0)),
            float(record.get("first_person_stamp", 0.0)),
            record.get("overlay_crop") or record.get("overlay") or "",
            record.get("first_person") or "",
        )
        record["panel"] = panel_path

    def rosout_callback(self, msg: Log) -> None:
        if self.shutting_down:
            return
        if getattr(msg, "name", "") != "/move_base":
            return
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
        line = (
            f"[{stamp:.6f}] level={int(msg.level)} "
            f"name={msg.name} file={msg.file} function={msg.function} line={msg.line} "
            f"msg={msg.msg}\n"
        )
        with self.lock:
            self.move_base_log_file.write(line)

    def explore_status_callback(self, msg: String) -> None:
        if self.shutting_down:
            return
        now = time.time()
        try:
            payload = json.loads(msg.data)
        except ValueError:
            payload = {"raw": msg.data}
        active_goal = None
        state = payload.get("state") if isinstance(payload, dict) else None
        if isinstance(state, dict):
            active_goal = state.get("active_goal")
        with self.lock:
            active_changed = active_goal != self.last_explore_active_goal
            if now - self.last_explore_status_time < self.args.explore_status_period_sec and not active_changed:
                return
            self.last_explore_status_time = now
            self.last_explore_active_goal = active_goal
            self._write_event("explore_status", {"elapsed_sec": now - self.start_wall_time, "payload": payload})

    def _analyze_goal(
        self,
        grid: OccupancyGrid | None,
        pose: tuple[float, float, float] | None,
        goal_xy: tuple[float, float],
    ) -> dict:
        robot_distance = None
        if pose is not None:
            robot_distance = math.hypot(goal_xy[0] - pose[0], goal_xy[1] - pose[1])
        result = {
            "robot_distance_m": robot_distance,
            "goal_cell": None,
            "goal_cell_value": None,
            "goal_is_free": False,
            "unknown_cells_near_goal": 0,
            "free_cells_near_goal": 0,
            "occupied_cells_near_goal": 0,
            "nearest_unknown_m": None,
            "frontier_like": False,
        }
        if grid is None:
            return result
        goal_cell = _world_to_cell(grid, goal_xy[0], goal_xy[1])
        result["goal_cell"] = list(goal_cell) if goal_cell is not None else None
        goal_value = _grid_value(grid, goal_cell)
        result["goal_cell_value"] = goal_value
        result["goal_is_free"] = _is_free(goal_value)
        if goal_cell is None:
            return result

        resolution = float(grid.info.resolution)
        radius_cells = max(1, int(math.ceil(self.args.frontier_check_radius_m / max(resolution, 1e-6))))
        nearest_unknown_cells = None
        width = int(grid.info.width)
        height = int(grid.info.height)
        gx, gy = goal_cell
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > radius_cells * radius_cells:
                    continue
                cx = gx + dx
                cy = gy + dy
                if cx < 0 or cy < 0 or cx >= width or cy >= height:
                    continue
                value = int(grid.data[cy * width + cx])
                if value < 0:
                    result["unknown_cells_near_goal"] += 1
                    dist_cells = math.hypot(dx, dy)
                    nearest_unknown_cells = dist_cells if nearest_unknown_cells is None else min(nearest_unknown_cells, dist_cells)
                elif value <= 20:
                    result["free_cells_near_goal"] += 1
                elif value >= 50:
                    result["occupied_cells_near_goal"] += 1
        if nearest_unknown_cells is not None:
            result["nearest_unknown_m"] = nearest_unknown_cells * resolution
        result["frontier_like"] = bool(
            result["goal_is_free"]
            and result["unknown_cells_near_goal"] > 0
            and result["nearest_unknown_m"] is not None
            and result["nearest_unknown_m"] <= self.args.frontier_check_radius_m
        )
        return result

    def _render_overlay(
        self,
        path: Path,
        grid: OccupancyGrid,
        pose: tuple[float, float, float] | None,
        goal_xy: tuple[float, float],
        trajectory: list[tuple[float, float, float, float]],
        global_plan: dict | None = None,
        local_plan: dict | None = None,
    ) -> str:
        width = int(grid.info.width)
        height = int(grid.info.height)
        rgb = bytearray(width * height * 3)
        for y in range(height):
            for x in range(width):
                value = int(grid.data[y * width + x])
                if value < 0:
                    color = (178, 178, 178)
                elif value <= 20:
                    color = (248, 248, 245)
                elif value >= 50:
                    color = (28, 30, 32)
                else:
                    color = (118, 118, 118)
                py = height - 1 - y
                index = (py * width + x) * 3
                rgb[index : index + 3] = bytes(color)

        def cell_to_pixel(cell: tuple[int, int] | None) -> tuple[int, int] | None:
            if cell is None:
                return None
            return cell[0], height - 1 - cell[1]

        def draw_pixel(px: int, py: int, color: tuple[int, int, int]) -> None:
            if px < 0 or py < 0 or px >= width or py >= height:
                return
            index = (py * width + px) * 3
            rgb[index : index + 3] = bytes(color)

        def draw_circle(px: int, py: int, radius: int, color: tuple[int, int, int]) -> None:
            rr = radius * radius
            for oy in range(-radius, radius + 1):
                for ox in range(-radius, radius + 1):
                    if ox * ox + oy * oy <= rr:
                        draw_pixel(px + ox, py + oy, color)

        def draw_cross(px: int, py: int, radius: int, color: tuple[int, int, int]) -> None:
            for delta in range(-radius, radius + 1):
                draw_pixel(px + delta, py, color)
                draw_pixel(px, py + delta, color)

        def draw_polyline(points_world: list[tuple[float, float, float]] | list[tuple[float, float]], color: tuple[int, int, int]) -> None:
            pixels = []
            for point in points_world:
                world_x = point[0]
                world_y = point[1]
                pixel = cell_to_pixel(_world_to_cell(grid, world_x, world_y))
                if pixel is not None:
                    pixels.append(pixel)
            for start, end in zip(pixels, pixels[1:]):
                for px, py in _bresenham(start[0], start[1], end[0], end[1]):
                    draw_pixel(px, py, color)

        trajectory_pixels = []
        for _, x, y, _ in trajectory:
            pixel = cell_to_pixel(_world_to_cell(grid, x, y))
            if pixel is not None:
                trajectory_pixels.append(pixel)
        for start, end in zip(trajectory_pixels, trajectory_pixels[1:]):
            for px, py in _bresenham(start[0], start[1], end[0], end[1]):
                draw_pixel(px, py, (20, 118, 230))

        if global_plan is not None:
            draw_polyline(global_plan.get("poses", []), (40, 190, 60))
        if local_plan is not None:
            draw_polyline(local_plan.get("poses", []), (240, 150, 20))

        if pose is not None:
            robot_pixel = cell_to_pixel(_world_to_cell(grid, pose[0], pose[1]))
            if robot_pixel is not None:
                draw_circle(robot_pixel[0], robot_pixel[1], 4, (0, 88, 255))
                heading_len = 10
                hx = int(round(robot_pixel[0] + math.cos(pose[2]) * heading_len))
                hy = int(round(robot_pixel[1] - math.sin(pose[2]) * heading_len))
                for px, py in _bresenham(robot_pixel[0], robot_pixel[1], hx, hy):
                    draw_pixel(px, py, (0, 45, 180))

        goal_pixel = cell_to_pixel(_world_to_cell(grid, goal_xy[0], goal_xy[1]))
        if goal_pixel is not None:
            draw_circle(goal_pixel[0], goal_pixel[1], 5, (230, 30, 45))
            draw_cross(goal_pixel[0], goal_pixel[1], 8, (150, 0, 20))

        _write_png(path, width, height, rgb)
        cropped = _crop_rgb_to_content(rgb, width, height, margin_px=self.args.crop_margin_px)
        if cropped is not None:
            crop_width, crop_height, crop_rgb = cropped
            crop_width, crop_height, crop_rgb = _scale_rgb_nearest(crop_rgb, crop_width, crop_height, self.args.crop_scale)
            crop_path = path.with_name(f"{path.stem}_crop.png")
            _write_png(crop_path, crop_width, crop_height, crop_rgb)
            return str(crop_path)
        return ""

    def _render_subgoal_panel(
        self,
        index: int,
        elapsed_sec: float,
        goal_stamp: float,
        image_stamp: float,
        overlay_path: str,
        first_person_path: str,
    ) -> str:
        if not overlay_path:
            return ""
        left = _read_png(Path(overlay_path))
        if left is None:
            return ""
        right = _read_png(Path(first_person_path)) if first_person_path else None
        title = f"#{index:03d} T={elapsed_sec:.1f}S GOAL={goal_stamp:.3f} IMG={image_stamp:.3f}"
        panel_width, panel_height, panel_rgb = _make_side_by_side_panel(
            left,
            right,
            title=title,
            image_height=self.args.panel_image_height_px,
            gap_px=self.args.panel_gap_px,
            title_height_px=self.args.panel_title_height_px,
        )
        panel_path = self.panel_dir / f"subgoal_{index:04d}_panel.png"
        _write_png(panel_path, panel_width, panel_height, panel_rgb)
        return str(panel_path)

    def _render_subgoal_contact_sheet(self) -> str:
        panels = []
        for record in self.subgoal_records:
            panel_path = record.get("panel") if isinstance(record, dict) else ""
            if not panel_path:
                continue
            loaded = _read_png(Path(panel_path))
            if loaded is not None:
                panels.append(loaded)
        contact_sheet = _make_contact_sheet(
            panels,
            columns=self.args.contact_sheet_columns,
            gap_px=self.args.contact_sheet_gap_px,
        )
        if contact_sheet is None:
            return ""
        width, height, rgb = contact_sheet
        path = self.output_dir / "subgoal_panels_contact_sheet.png"
        _write_png(path, width, height, rgb)
        return str(path)

    def _write_event(self, event_type: str, payload: dict) -> None:
        row = {
            "type": event_type,
            "wall_time": time.time(),
            "elapsed_sec": time.time() - self.start_wall_time,
            **payload,
        }
        self.events_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def shutdown(self) -> None:
        with self.lock:
            if self.shutting_down:
                return
            self.shutting_down = True
            final_overlay = ""
            final_overlay_crop = ""
            if self.latest_grid is not None:
                final_overlay = str(self.output_dir / "final_map_trajectory.png")
                final_overlay_crop = self._render_overlay(
                    Path(final_overlay),
                    self.latest_grid,
                    self.latest_pose,
                    self.latest_pose[:2] if self.latest_pose else (0.0, 0.0),
                    list(self.trajectory),
                    global_plan=self.latest_global_plan,
                    local_plan=self.latest_local_plan,
                )
            subgoal_contact_sheet = self._render_subgoal_contact_sheet()
            summary = {
                "duration_sec": time.time() - self.start_wall_time,
                "distance_m": self.distance_m,
                "trajectory_samples": len(self.trajectory),
                "subgoal_count": self.goal_count,
                "current_subgoal_count": self.current_subgoal_count,
                "status_counts": self.status_counts,
                "cmd_vel_counts": self.cmd_vel_counts,
                "cmd_vel_nonzero_counts": self.cmd_vel_nonzero_counts,
                "cmd_vel_max_speed": self.cmd_vel_max_speed,
                "plan_message_counts": self.plan_message_counts,
                "first_pose": list(self.trajectory[0][1:]) if self.trajectory else None,
                "last_pose": list(self.trajectory[-1][1:]) if self.trajectory else None,
                "subgoals": self.subgoal_records,
                "final_overlay": final_overlay,
                "final_overlay_crop": final_overlay_crop,
                "subgoal_contact_sheet": subgoal_contact_sheet,
                "move_base_plan_csv": str(self.output_dir / "move_base_plans.csv"),
                "move_base_rosout_log": str(self.output_dir / "move_base_rosout.log"),
            }
            (self.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            self._write_event("recorder_shutdown", summary)
            for handle in [
                self.events_file,
                self.trajectory_file,
                self.subgoals_file,
                self.status_file,
                self.cmd_vel_file,
                self.plan_file,
                self.move_base_log_file,
            ]:
                handle.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record explore_py runtime debug artifacts.")
    parser.add_argument("--output-dir", default="", help="Directory for JSONL, CSV, and PNG overlays.")
    parser.add_argument("--occupancy-grid-topic", default="/struct_mapping/occ_map")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--goal-topic", default="/move_base_simple/goal")
    parser.add_argument("--current-subgoal-topic", default="/explore_py/current_subgoal")
    parser.add_argument("--move-base-status-topic", default="/move_base/status")
    parser.add_argument("--explore-status-topic", default="/explore_py/status")
    parser.add_argument("--global-plan-topic", default="/move_base/GlobalPlanner/plan")
    parser.add_argument("--local-plan-topic", default="/move_base/DWAPlannerROS/local_plan")
    parser.add_argument("--rosout-topic", default="/rosout_agg")
    parser.add_argument("--image-topic", default="/molmo_spaces/head_camera/image")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--cmd-vel-stamped-topic", default="/cmd_vel_stamped")
    parser.add_argument("--frontier-check-radius-m", type=float, default=1.0)
    parser.add_argument("--trajectory-period-sec", type=float, default=0.5)
    parser.add_argument("--trajectory-min-step-m", type=float, default=0.02)
    parser.add_argument("--explore-status-period-sec", type=float, default=2.0)
    parser.add_argument("--max-odom-jump-m", type=float, default=3.0)
    parser.add_argument("--crop-margin-px", type=int, default=40)
    parser.add_argument("--crop-scale", type=int, default=4)
    parser.add_argument("--panel-image-height-px", type=int, default=520)
    parser.add_argument("--panel-title-height-px", type=int, default=34)
    parser.add_argument("--panel-gap-px", type=int, default=12)
    parser.add_argument("--contact-sheet-columns", type=int, default=1)
    parser.add_argument("--contact-sheet-gap-px", type=int, default=16)
    parser.add_argument("--cmd-vel-record-period-sec", type=float, default=0.2)
    parser.add_argument("--cmd-vel-nonzero-threshold", type=float, default=1e-4)
    return parser.parse_args(rospy.myargv()[1:])


def main() -> None:
    args = _parse_args()
    rospy.init_node("explore_py_debug_recorder")
    default_dir = Path("/tmp/explore_py_runs") / time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or rospy.get_param("~output_dir", str(default_dir))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    recorder = ExploreDebugRecorder(output_dir, args)
    rospy.loginfo("[explore_py_debug_recorder] writing artifacts to %s", output_dir)
    rospy.spin()
    # Keep a strong reference until shutdown hooks finish.
    _ = recorder


if __name__ == "__main__":
    main()
