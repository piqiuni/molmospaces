#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
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
import tf
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PointStamped, PoseStamped, Twist, TwistStamped
from map_msgs.msg import OccupancyGridUpdate
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from rosgraph_msgs.msg import Log
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - optional debug dependency
    cv2 = None
    np = None


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


def _step4(value: int | float | None) -> str:
    return f"{int(value or 0):04d}"


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


def _is_occupied(value: int | None) -> bool:
    return value is not None and value >= 50


def _is_frontier_cell_data(data, width: int, height: int, x: int, y: int) -> bool:
    if x < 0 or y < 0 or x >= width or y >= height:
        return False
    if not _is_free(int(data[y * width + x])):
        return False
    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
        if nx < 0 or ny < 0 or nx >= width or ny >= height:
            continue
        if _is_unknown(int(data[ny * width + nx])):
            return True
    return False


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


def _write_grid_pgm_yaml(prefix: Path, grid: OccupancyGrid) -> None:
    width = int(grid.info.width)
    height = int(grid.info.height)
    pgm_path = prefix.with_suffix(".pgm")
    yaml_path = prefix.with_suffix(".yaml")
    raw = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            value = int(grid.data[y * width + x])
            if value < 0:
                pixel = 205
            elif value >= 50:
                pixel = 0
            else:
                pixel = 254
            py = height - 1 - y
            raw[py * width + x] = pixel
    pgm_path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + bytes(raw))
    origin = grid.info.origin
    yaml_path.write_text(
        "\n".join(
            [
                f"image: {pgm_path}",
                f"resolution: {float(grid.info.resolution):.6f}",
                "origin: "
                f"[{float(origin.position.x):.6f}, {float(origin.position.y):.6f}, {_grid_origin_yaw(grid):.6f}]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.196",
                "",
            ]
        )
    )


def _copy_grid(grid: OccupancyGrid) -> OccupancyGrid:
    copied = OccupancyGrid()
    copied.header = grid.header
    copied.info = grid.info
    copied.data = list(grid.data)
    return copied


def _apply_grid_update(grid: OccupancyGrid, update: OccupancyGridUpdate) -> bool:
    width = int(grid.info.width)
    height = int(grid.info.height)
    ux = int(update.x)
    uy = int(update.y)
    uw = int(update.width)
    uh = int(update.height)
    if width <= 0 or height <= 0 or uw <= 0 or uh <= 0:
        return False
    if ux < 0 or uy < 0 or ux + uw > width or uy + uh > height:
        return False
    if len(update.data) < uw * uh:
        return False
    data = list(grid.data)
    for row in range(uh):
        src = row * uw
        dst = (uy + row) * width + ux
        data[dst : dst + uw] = update.data[src : src + uw]
    grid.data = data
    grid.header.stamp = update.header.stamp
    return True


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


def _content_bbox(
    rgb: bytearray,
    width: int,
    height: int,
    background: tuple[int, int, int] = (178, 178, 178),
    ignore_top_px: int = 0,
) -> tuple[int, int, int, int] | None:
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    bg = bytes(background)
    for y in range(max(0, int(ignore_top_px)), height):
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
    return min_x, min_y, max_x, max_y


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    margin_px: int,
) -> tuple[int, int, int, int]:
    min_x, min_y, max_x, max_y = bbox
    return (
        max(0, min_x - margin_px),
        max(0, min_y - margin_px),
        min(width - 1, max_x + margin_px),
        min(height - 1, max_y + margin_px),
    )


def _union_bbox(
    first: tuple[int, int, int, int] | None,
    second: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if first is None:
        return second
    if second is None:
        return first
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _crop_rgb_to_bbox(
    rgb: bytearray,
    width: int,
    height: int,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, bytearray]:
    min_x, min_y, max_x, max_y = bbox
    min_x = max(0, min(width - 1, min_x))
    min_y = max(0, min(height - 1, min_y))
    max_x = max(min_x, min(width - 1, max_x))
    max_y = max(min_y, min(height - 1, max_y))
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


def _make_titled_panel(
    image: tuple[int, int, bytearray],
    title: str,
    title_height_px: int,
    background: tuple[int, int, int] = (255, 255, 255),
) -> tuple[int, int, bytearray]:
    width, height, rgb = image
    title_height_px = max(0, int(title_height_px))
    canvas = bytearray(list(background) * width * (height + title_height_px))
    if title_height_px > 0:
        _draw_text(canvas, width, height + title_height_px, 12, 8, title, (20, 20, 20), scale=2)
    _paste_rgb(canvas, width, height + title_height_px, rgb, width, height, 0, title_height_px)
    return width, height + title_height_px, canvas


def _make_contact_sheet(
    panels: list[tuple[int, int, bytearray]],
    columns: int,
    gap_px: int,
    background: tuple[int, int, int] = (245, 245, 242),
) -> tuple[int, int, bytearray] | None:
    if not panels:
        return None
    columns = max(1, int(columns))
    panel_rows = [panels[index : index + columns] for index in range(0, len(panels), columns)]
    row_widths = [
        sum(panel_width for panel_width, _, _ in row) + max(0, len(row) - 1) * gap_px
        for row in panel_rows
    ]
    row_heights = [max(panel_height for _, panel_height, _ in row) for row in panel_rows]
    width = max(row_widths)
    height = sum(row_heights) + max(0, len(panel_rows) - 1) * gap_px
    canvas = bytearray(list(background) * width * height)
    y = 0
    for row, row_height in zip(panel_rows, row_heights):
        x = 0
        for panel_width, panel_height, panel_rgb in row:
            _paste_rgb(canvas, width, height, panel_rgb, panel_width, panel_height, x, y)
            x += panel_width + gap_px
        y += row_height + gap_px
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
        self.uniform_overlay_dir = output_dir / "subgoal_overlays_uniform_crop"
        self.uniform_overlay_titled_dir = output_dir / "subgoal_overlays_uniform_titled"
        self.first_person_dir = output_dir / "first_person"
        self.external_dir = output_dir / "external_camera"
        self.video_dir = output_dir / "videos"
        self.video_camera_frame_dir = self.video_dir / "camera_frames"
        self.video_map_frame_dir = self.video_dir / "map_frames"
        self.video_global_costmap_frame_dir = self.video_dir / "global_costmap_frames"
        self.video_local_costmap_frame_dir = self.video_dir / "local_costmap_frames"
        self.video_semantic_spatial_frame_dir = self.video_dir / "semantic_spatial_frames"
        self.video_semantic_topology_frame_dir = self.video_dir / "semantic_topology_frames"
        self.video_composite_frame_dir = self.video_dir / "composite_frames"
        self.video_external_frame_dir = self.video_dir / "external_camera_frames"
        self.semantic_keyframe_dir = output_dir / "semantic_keyframes"
        self.graph_dir = output_dir / "graph"
        self.panel_dir = output_dir / "subgoal_panels"
        self.uniform_panel_dir = output_dir / "subgoal_panels_uniform_crop"
        self.stall_snapshot_dir = output_dir / "stall_snapshots"
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        self.uniform_overlay_dir.mkdir(parents=True, exist_ok=True)
        self.uniform_overlay_titled_dir.mkdir(parents=True, exist_ok=True)
        self.first_person_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.video_camera_frame_dir.mkdir(parents=True, exist_ok=True)
        self.video_map_frame_dir.mkdir(parents=True, exist_ok=True)
        self.video_global_costmap_frame_dir.mkdir(parents=True, exist_ok=True)
        self.video_local_costmap_frame_dir.mkdir(parents=True, exist_ok=True)
        self.video_semantic_spatial_frame_dir.mkdir(parents=True, exist_ok=True)
        self.video_semantic_topology_frame_dir.mkdir(parents=True, exist_ok=True)
        self.video_composite_frame_dir.mkdir(parents=True, exist_ok=True)
        self.video_external_frame_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_keyframe_dir.mkdir(parents=True, exist_ok=True)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.panel_dir.mkdir(parents=True, exist_ok=True)
        self.uniform_panel_dir.mkdir(parents=True, exist_ok=True)
        self.stall_snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.args = args
        self.tf_listener = tf.TransformListener()
        self.lock = threading.RLock()
        self.video_lock = threading.RLock()
        self.shutting_down = False
        self.start_wall_time = time.time()
        self.latest_grid: OccupancyGrid | None = None
        self.latest_global_costmap: OccupancyGrid | None = None
        self.latest_local_costmap: OccupancyGrid | None = None
        self.latest_grid_video_rgb = None
        self.latest_global_costmap_video_rgb = None
        self.latest_local_costmap_video_rgb = None
        self.latest_grid_video_stamp = 0.0
        self.latest_global_costmap_video_stamp = 0.0
        self.latest_local_costmap_video_stamp = 0.0
        self.latest_grid_wall_time = 0.0
        self.latest_global_costmap_wall_time = 0.0
        self.latest_local_costmap_wall_time = 0.0
        self.latest_grid_step = 0
        self.latest_global_costmap_step = 0
        self.latest_local_costmap_step = 0
        self.latest_image: tuple[float, int, int, bytearray] | None = None
        self.latest_image_step = 0
        self.latest_external_image: tuple[float, int, int, bytearray] | None = None
        self.latest_external_image_step = 0
        self.last_image_wall_time = 0.0
        self.final_first_person_path = ""
        self.latest_pose: tuple[float, float, float] | None = None
        self.latest_pose_step = 0
        self.latest_global_plan: dict | None = None
        self.latest_local_global_plan: dict | None = None
        self.latest_local_plan: dict | None = None
        self.latest_gt_observations: dict = {}
        self.recording_episode_id = ""
        self.observed_instance_ids: set[str] = set()
        self.latest_unified_graph: dict = {}
        self.previous_unified_graph: dict = {}
        self.semantic_events: list[dict] = []
        self.pending_semantic_keyframe_revision = -1
        self.last_semantic_keyframe_revision = -1
        self.topology_slots: dict[str, tuple[int, int]] = {}
        self.topology_next_slot = {"room": 0, "portal": 0, "container": 0, "object": 0}
        self.last_odom_xy: tuple[float, float] | None = None
        self.last_recorded_odom_time = 0.0
        self.stall_reference_xy: tuple[float, float] | None = None
        self.stall_reference_yaw: float | None = None
        self.stall_reference_time = 0.0
        self.last_stall_snapshot_time = 0.0
        self.stall_snapshot_count = 0
        self.stall_snapshot_records: list[dict] = []
        self.distance_m = 0.0
        self.trajectory: list[tuple[float, float, float, float]] = []
        self.goal_count = 0
        self.debug_step = 0
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
        self.plan_message_counts = {"global": 0, "local_global": 0, "local": 0}
        self.plan_records = {"global": [], "local_global": [], "local": []}
        self.subgoal_records: list[dict] = []
        video_stem = "overview_6panel" if args.semantic_video else "first_person"
        self.first_person_video_path = str(self.video_dir / f"{video_stem}.mp4")
        self.first_person_video_raw_path = str(self.video_dir / f"{video_stem}_raw.mp4")
        self.first_person_video_writer = None
        self.first_person_video_size: tuple[int, int] | None = None
        self.first_person_video_frame_count = 0
        self.first_person_video_frames: list[dict] = []
        self.last_first_person_video_frame_time = 0.0
        self.video_map_bbox: tuple[int, int, int, int] | None = None
        self.video_global_costmap_bbox: tuple[int, int, int, int] | None = None
        self.video_local_costmap_bbox: tuple[int, int, int, int] | None = None
        self.first_person_video_error = ""
        self.first_person_video_codec_name = "h264" if args.first_person_video_h264 else str(args.first_person_video_codec)
        self.external_video_path = str(self.video_dir / "external_camera.mp4")
        self.external_video_raw_path = str(self.video_dir / "external_camera_raw.mp4")
        self.external_video_frame_count = 0
        self.external_video_frames: list[dict] = []
        self.last_external_video_frame_time = 0.0
        self.external_video_error = ""
        self.external_video_codec_name = "h264" if args.first_person_video_h264 else str(args.first_person_video_codec)
        self.subscribers = []
        if args.first_person_video and (cv2 is None or np is None):
            self.first_person_video_error = "cv2_or_numpy_unavailable"
            self._write_optional_dependency_warning()

        self.events_file = (output_dir / "events.jsonl").open("a", buffering=1)
        self.trajectory_file = (output_dir / "trajectory.csv").open("a", newline="", buffering=1)
        self.subgoals_file = (output_dir / "subgoals.csv").open("a", newline="", buffering=1)
        self.status_file = (output_dir / "move_base_status.csv").open("a", newline="", buffering=1)
        self.cmd_vel_file = (output_dir / "cmd_vel.csv").open("a", newline="", buffering=1)
        self.plan_file = (output_dir / "move_base_plans.csv").open("a", newline="", buffering=1)
        self.move_base_log_file = (output_dir / "move_base_rosout.log").open("a", buffering=1)
        self.semantic_events_file = (self.graph_dir / "graph_revision_events.jsonl").open("a", buffering=1)
        self.video_frames_file = (output_dir / "video_frames.csv").open("a", newline="", buffering=1)

        self.trajectory_writer = csv.DictWriter(
            self.trajectory_file,
            fieldnames=["step_id", "elapsed_sec", "stamp", "x", "y", "yaw", "step_distance_m", "total_distance_m"],
        )
        self.subgoals_writer = csv.DictWriter(
            self.subgoals_file,
            fieldnames=[
                "index",
                "step_id",
                "grid_step",
                "image_step",
                "elapsed_sec",
                "stamp",
                "x",
                "y",
                "yaw",
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
                "external_image",
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
            fieldnames=["step_id", "elapsed_sec", "stamp", "goal_id", "status", "status_name", "text"],
        )
        self.cmd_vel_writer = csv.DictWriter(
            self.cmd_vel_file,
            fieldnames=["step_id", "elapsed_sec", "stamp", "topic", "linear_x", "linear_y", "angular_z", "speed"],
        )
        self.plan_writer = csv.DictWriter(
            self.plan_file,
            fieldnames=[
                "step_id",
                "elapsed_sec",
                "topic",
                "plan_type",
                "message_index",
                "stamp",
                "frame_id",
                "pose_index",
                "pose_count",
                "x",
                "y",
                "yaw",
            ],
        )
        self.video_frames_writer = csv.DictWriter(
            self.video_frames_file,
            fieldnames=[
                "frame_index",
                "step_id",
                "elapsed_sec",
                "image_stamp",
                "map_stamp",
                "stamp_delta_sec",
                "map_sync",
                "map_fresh",
                "map_age_wall_sec",
                "distance_m",
                "goal_count",
                "robot_x",
                "robot_y",
                "robot_yaw",
                "active_goal",
                "stuck_state",
                "stuck_duration_sec",
                "stuck_moved_m",
                "stuck_yaw_delta_rad",
                "camera_frame",
                "map_frame",
                "global_costmap_step",
                "local_costmap_step",
                "global_costmap_frame",
                "local_costmap_frame",
                "semantic_spatial_frame",
                "semantic_topology_frame",
                "composite_frame",
            ],
        )
        self.trajectory_writer.writeheader()
        self.subgoals_writer.writeheader()
        self.status_writer.writeheader()
        self.cmd_vel_writer.writeheader()
        self.plan_writer.writeheader()
        self.video_frames_writer.writeheader()

        self.subscribers.append(rospy.Subscriber(args.occupancy_grid_topic, OccupancyGrid, self.occupancy_callback, queue_size=1))
        self.subscribers.append(rospy.Subscriber(args.global_costmap_topic, OccupancyGrid, self.global_costmap_callback, queue_size=1))
        self.subscribers.append(rospy.Subscriber(args.global_costmap_updates_topic, OccupancyGridUpdate, self.global_costmap_update_callback, queue_size=20))
        self.subscribers.append(rospy.Subscriber(args.local_costmap_topic, OccupancyGrid, self.local_costmap_callback, queue_size=1))
        self.subscribers.append(rospy.Subscriber(args.local_costmap_updates_topic, OccupancyGridUpdate, self.local_costmap_update_callback, queue_size=50))
        self.subscribers.append(rospy.Subscriber(args.image_topic, Image, self.image_callback, queue_size=1))
        if args.external_image_topic:
            self.subscribers.append(rospy.Subscriber(args.external_image_topic, Image, self.external_image_callback, queue_size=1))
        self.subscribers.append(rospy.Subscriber(args.odom_topic, Odometry, self.odom_callback, queue_size=50))
        self.subscribers.append(rospy.Subscriber(args.goal_topic, PoseStamped, self.goal_callback, queue_size=20))
        self.subscribers.append(rospy.Subscriber(args.current_subgoal_topic, PointStamped, self.current_subgoal_callback, queue_size=20))
        self.subscribers.append(rospy.Subscriber(args.move_base_status_topic, GoalStatusArray, self.move_base_status_callback, queue_size=20))
        self.subscribers.append(rospy.Subscriber(args.explore_status_topic, String, self.explore_status_callback, queue_size=20))
        self.subscribers.append(rospy.Subscriber(args.gt_observations_topic, String, self.gt_observations_callback, queue_size=2))
        self.subscribers.append(rospy.Subscriber(args.unified_graph_topic, String, self.unified_graph_callback, queue_size=2))
        self.subscribers.append(rospy.Subscriber(args.cmd_vel_topic, Twist, self.cmd_vel_callback, callback_args=args.cmd_vel_topic, queue_size=50))
        self.subscribers.append(rospy.Subscriber(args.global_plan_topic, NavPath, self.plan_callback, callback_args=("global", args.global_plan_topic), queue_size=20))
        self.subscribers.append(rospy.Subscriber(args.local_global_plan_topic, NavPath, self.plan_callback, callback_args=("local_global", args.local_global_plan_topic), queue_size=20))
        self.subscribers.append(rospy.Subscriber(args.local_plan_topic, NavPath, self.plan_callback, callback_args=("local", args.local_plan_topic), queue_size=20))
        self.subscribers.append(rospy.Subscriber(args.rosout_topic, Log, self.rosout_callback, queue_size=200))
        self.subscribers.append(rospy.Subscriber(
            args.cmd_vel_stamped_topic,
            TwistStamped,
            self.cmd_vel_stamped_callback,
            callback_args=args.cmd_vel_stamped_topic,
            queue_size=50,
        ))
        self.stall_timer = rospy.Timer(rospy.Duration(max(0.5, args.stall_check_period_sec)), self.stall_timer_callback)
        rospy.on_shutdown(self.shutdown)

    def occupancy_callback(self, msg: OccupancyGrid) -> None:
        with self.lock:
            if self.shutting_down:
                return
            self.latest_grid = msg
            self.latest_grid_video_rgb = self._grid_to_video_rgb_locked(msg)
            self.latest_grid_video_stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
            self.latest_grid_wall_time = time.time()
            self.latest_grid_step = self.debug_step

    def gt_observations_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        with self.lock:
            if self.shutting_down:
                return
            episode_id = str(payload.get("episode_id") or "")
            if bool(payload.get("episode_reset")) or (episode_id and episode_id != self.recording_episode_id):
                self.recording_episode_id = episode_id
                self.observed_instance_ids.clear()
                self.topology_slots.clear()
                self.topology_next_slot = {"room": 0, "portal": 0, "container": 0, "object": 0}
            for observation in payload.get("observations") or []:
                instance_id = str(observation.get("instance_id") or "")
                if instance_id:
                    self.observed_instance_ids.add(instance_id)
            self.latest_gt_observations = payload

    def unified_graph_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        with self.lock:
            if self.shutting_down:
                return
            revision = int(payload.get("graph_revision", 0) or 0)
            events = self._semantic_graph_delta(self.latest_unified_graph, payload)
            self.previous_unified_graph = self.latest_unified_graph
            self.latest_unified_graph = payload
            (self.graph_dir / "graph_latest.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            )
            for event in events:
                row = {
                    "wall_time": time.time(),
                    "elapsed_sec": time.time() - self.start_wall_time,
                    "step_id": self.debug_step,
                    "graph_revision": revision,
                    **event,
                }
                self.semantic_events.append(row)
                self.semantic_events = self.semantic_events[-50:]
                self.semantic_events_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            if events:
                self.pending_semantic_keyframe_revision = revision

    @staticmethod
    def _semantic_graph_delta(previous: dict, current: dict) -> list[dict]:
        previous_nodes = {node.get("id"): node for node in previous.get("nodes", []) if node.get("id")}
        current_nodes = {node.get("id"): node for node in current.get("nodes", []) if node.get("id")}
        previous_edges = {edge.get("id"): edge for edge in previous.get("edges", []) if edge.get("id")}
        current_edges = {edge.get("id"): edge for edge in current.get("edges", []) if edge.get("id")}
        events = []
        for node_id in sorted(current_nodes.keys() - previous_nodes.keys()):
            node = current_nodes[node_id]
            events.append({"event": "NEW_NODE", "node_id": node_id, "label": node.get("label", "")})
        for edge_id in sorted(current_edges.keys() - previous_edges.keys()):
            edge = current_edges[edge_id]
            events.append(
                {
                    "event": "NEW_EDGE",
                    "edge_id": edge_id,
                    "relation": edge.get("relation", ""),
                    "src_id": edge.get("src_id", ""),
                    "dst_id": edge.get("dst_id", ""),
                }
            )
        for node_id in sorted(current_nodes.keys() & previous_nodes.keys()):
            before = previous_nodes[node_id]
            after = current_nodes[node_id]
            before_state = (before.get("interaction") or {}).get("state", "unknown")
            after_state = (after.get("interaction") or {}).get("state", "unknown")
            if before_state != after_state:
                events.append(
                    {
                        "event": "STATE_CHANGED",
                        "node_id": node_id,
                        "before": before_state,
                        "after": after_state,
                    }
                )
        return events

    def global_costmap_callback(self, msg: OccupancyGrid) -> None:
        with self.lock:
            if self.shutting_down:
                return
            self.latest_global_costmap = _copy_grid(msg)
            self.latest_global_costmap_video_rgb = self._grid_to_video_rgb_locked(msg)
            self.latest_global_costmap_video_stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
            self.latest_global_costmap_wall_time = time.time()
            self.latest_global_costmap_step = self.debug_step

    def global_costmap_update_callback(self, msg: OccupancyGridUpdate) -> None:
        with self.lock:
            if self.shutting_down or self.latest_global_costmap is None:
                return
            if not _apply_grid_update(self.latest_global_costmap, msg):
                return
            self.latest_global_costmap_video_rgb = self._grid_to_video_rgb_locked(self.latest_global_costmap)
            self.latest_global_costmap_video_stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
            self.latest_global_costmap_wall_time = time.time()
            self.latest_global_costmap_step = self.debug_step

    def local_costmap_callback(self, msg: OccupancyGrid) -> None:
        with self.lock:
            if self.shutting_down:
                return
            self.latest_local_costmap = _copy_grid(msg)
            self.latest_local_costmap_video_rgb = self._grid_to_video_rgb_locked(msg)
            self.latest_local_costmap_video_stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
            self.latest_local_costmap_wall_time = time.time()
            self.latest_local_costmap_step = self.debug_step

    def local_costmap_update_callback(self, msg: OccupancyGridUpdate) -> None:
        with self.lock:
            # Local costmap uses a rolling window. OccupancyGridUpdate does not
            # carry the updated origin, so applying patches locally can make the
            # debug image drift away from the true costmap frame. Use the full
            # /costmap message for local visualization instead.
            return

    def image_callback(self, msg: Image) -> None:
        if self.shutting_down:
            return
        converted = _image_msg_to_rgb(msg)
        if converted is None:
            return
        width, height, rgb = converted
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
        with self.lock:
            if self.shutting_down:
                return
            self.debug_step += 1
            self.latest_image = (stamp, width, height, rgb)
            self.latest_image_step = self.debug_step
            self.last_image_wall_time = time.time()
        with self.video_lock:
            if self.shutting_down:
                return
            self._record_first_person_video_frame_locked(width, height, rgb)

    def external_image_callback(self, msg: Image) -> None:
        if self.shutting_down:
            return
        converted = _image_msg_to_rgb(msg)
        if converted is None:
            return
        width, height, rgb = converted
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
        with self.lock:
            if self.shutting_down:
                return
            self.latest_external_image = (stamp, width, height, rgb)
            self.latest_external_image_step = self.debug_step
        with self.video_lock:
            if self.shutting_down:
                return
            self._record_external_video_frame_locked(width, height, rgb, stamp)

    def _write_optional_dependency_warning(self) -> None:
        sys.stderr.write(
            "[explore_py_debug_recorder] first-person video disabled: "
            "cv2/numpy is unavailable in this Python environment.\n"
        )

    def _draw_gt_observations_locked(self, frame, source_width: int, source_height: int) -> None:
        observations = list(self.latest_gt_observations.get("observations") or [])
        scale_x = float(frame.shape[1]) / max(1, source_width)
        scale_y = float(frame.shape[0]) / max(1, source_height)
        for observation in observations:
            bbox = observation.get("bbox_2d") or []
            image_size = observation.get("image_size") or [source_width, source_height]
            if len(bbox) != 4 or len(image_size) != 2:
                continue
            bbox_scale_x = float(frame.shape[1]) / max(1, int(image_size[0]))
            bbox_scale_y = float(frame.shape[0]) / max(1, int(image_size[1]))
            x0, y0, x1, y1 = [int(value) for value in bbox]
            start = (int(x0 * bbox_scale_x), int(y0 * bbox_scale_y))
            end = (int(x1 * bbox_scale_x), int(y1 * bbox_scale_y))
            is_door = bool(observation.get("is_door"))
            is_container = bool(observation.get("is_receptacle"))
            color = (238, 80, 50) if is_door else (170, 70, 220) if is_container else (20, 210, 210)
            cv2.rectangle(frame, start, end, color, 2, cv2.LINE_AA)
            label = f"{observation.get('semantic_name', 'obj')} {observation.get('instance_id', '')}"
            joint_value = observation.get("joint_value")
            if joint_value is not None and observation.get("primary_joint_name"):
                label += f" q={float(joint_value):.2f}"
            label_y = max(18, start[1] - 5)
            cv2.putText(frame, label[:48], (start[0], label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, max(0, frame.shape[0] - 28)), (min(frame.shape[1] - 1, 420), frame.shape[0] - 1), (255, 255, 255), -1)
        cv2.putText(
            frame,
            f"GT visible={len(observations)} frame={self.latest_gt_observations.get('frame_index', '-')} source=realtime_gt",
            (8, frame.shape[0] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _node_xy(node: dict) -> tuple[float, float] | None:
        center = node.get("aabb_center") or node.get("centroid") or []
        if len(center) < 2:
            return None
        return float(center[0]), float(center[1])

    def _node_observed_in_recording(self, node: dict) -> bool:
        if node.get("type") == "room":
            return True
        instance_id = str((node.get("attributes") or {}).get("instance_id") or "")
        return bool(instance_id and instance_id in self.observed_instance_ids)

    @staticmethod
    def _short_node_id(node: dict) -> str:
        instance_id = str((node.get("attributes") or {}).get("instance_id") or "")
        token = instance_id or str(node.get("id") or "")
        suffix = token.rsplit("_", 1)[-1]
        try:
            return f"#{int(suffix)}"
        except ValueError:
            return suffix[-6:]

    @staticmethod
    def _semantic_node_color(node: dict) -> tuple[int, int, int]:
        node_type = str(node.get("type", "object"))
        if node_type == "room":
            return (205, 225, 245)
        if node_type == "portal":
            state = str((node.get("interaction") or {}).get("state", "unknown"))
            return (50, 190, 70) if state in {"open", "ajar"} else (235, 70, 55) if state == "closed" else (235, 175, 45)
        if node_type == "container":
            return (175, 75, 220)
        if node_type == "support":
            return (50, 125, 220)
        return (30, 190, 195) if node.get("is_currently_visible") else (145, 145, 145)

    def _render_semantic_spatial_panel_locked(self, panel_width: int, panel_height: int, pose) -> object:
        panel = np.full((panel_height, panel_width, 3), 246, dtype=np.uint8)
        graph = self.latest_unified_graph
        nodes = [node for node in graph.get("nodes") or [] if self._node_observed_in_recording(node)]
        positions = [self._node_xy(node) for node in nodes]
        positions = [position for position in positions if position is not None]
        if pose is not None:
            positions.append((float(pose[0]), float(pose[1])))
        cv2.putText(panel, f"SEMANTIC XY rev={graph.get('graph_revision', 0)} nodes={len(nodes)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
        if not positions:
            cv2.putText(panel, "WAITING FOR UNIFIED GRAPH", (40, panel_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 90), 2, cv2.LINE_AA)
            return panel
        min_x = min(position[0] for position in positions)
        max_x = max(position[0] for position in positions)
        min_y = min(position[1] for position in positions)
        max_y = max(position[1] for position in positions)
        for node in nodes:
            position = self._node_xy(node)
            size = node.get("aabb_size") or [0.0, 0.0, 0.0]
            if position is None or len(size) < 2:
                continue
            min_x = min(min_x, position[0] - float(size[0]) * 0.5)
            max_x = max(max_x, position[0] + float(size[0]) * 0.5)
            min_y = min(min_y, position[1] - float(size[1]) * 0.5)
            max_y = max(max_y, position[1] + float(size[1]) * 0.5)
        margin = 38
        span_x = max(2.0, max_x - min_x)
        span_y = max(2.0, max_y - min_y)
        scale = min((panel_width - margin * 2) / span_x, (panel_height - margin * 2 - 20) / span_y)
        center_x = (min_x + max_x) * 0.5
        center_y = (min_y + max_y) * 0.5

        def to_px(x: float, y: float) -> tuple[int, int]:
            return int(panel_width * 0.5 + (x - center_x) * scale), int(panel_height * 0.53 - (y - center_y) * scale)

        for grid_x in range(math.floor(min_x), math.ceil(max_x) + 1):
            start = to_px(grid_x, min_y)
            end = to_px(grid_x, max_y)
            cv2.line(panel, start, end, (226, 226, 226), 1)
        for grid_y in range(math.floor(min_y), math.ceil(max_y) + 1):
            start = to_px(min_x, grid_y)
            end = to_px(max_x, grid_y)
            cv2.line(panel, start, end, (226, 226, 226), 1)
        node_lookup = {node.get("id"): node for node in nodes}
        for edge in graph.get("edges") or []:
            if edge.get("relation") not in {"connects", "contains", "supports"}:
                continue
            src = self._node_xy(node_lookup.get(edge.get("src_id"), {}))
            dst = self._node_xy(node_lookup.get(edge.get("dst_id"), {}))
            if src is None or dst is None:
                continue
            relation = edge.get("relation")
            color = (220, 90, 45) if relation == "connects" else (170, 75, 210) if relation == "contains" else (55, 125, 220)
            cv2.line(panel, to_px(*src), to_px(*dst), color, 2, cv2.LINE_AA)
        for node in sorted(nodes, key=lambda item: item.get("type") != "room"):
            position = self._node_xy(node)
            if position is None:
                continue
            size = node.get("aabb_size") or [0.25, 0.25, 0.0]
            half_w = max(3, int(max(0.08, float(size[0]) * 0.5) * scale))
            half_h = max(3, int(max(0.08, float(size[1]) * 0.5) * scale))
            center = to_px(*position)
            color = self._semantic_node_color(node)
            thickness = 2 if node.get("is_currently_visible") else 1
            if node.get("type") == "room":
                overlay = panel.copy()
                cv2.rectangle(overlay, (center[0] - half_w, center[1] - half_h), (center[0] + half_w, center[1] + half_h), color, -1)
                cv2.addWeighted(overlay, 0.35, panel, 0.65, 0, panel)
                cv2.rectangle(panel, (center[0] - half_w, center[1] - half_h), (center[0] + half_w, center[1] + half_h), (125, 150, 175), 1)
            else:
                cv2.rectangle(panel, (center[0] - half_w, center[1] - half_h), (center[0] + half_w, center[1] + half_h), color, thickness)
            short_id = self._short_node_id(node)
            if node.get("type") in {"room", "portal", "container"}:
                label = f"{short_id} {node.get('label', node.get('type', ''))}"[:22]
            else:
                label = short_id
            cv2.putText(panel, label, (center[0] + 3, center[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (35, 35, 35), 1, cv2.LINE_AA)
        if pose is not None:
            center = to_px(float(pose[0]), float(pose[1]))
            self._draw_cv_robot_arrow(panel, center, float(pose[2]), 14)
        return panel

    def _render_semantic_topology_panel_locked(self, panel_width: int, panel_height: int) -> object:
        panel = np.full((panel_height, panel_width, 3), 250, dtype=np.uint8)
        graph = self.latest_unified_graph
        all_nodes = [node for node in graph.get("nodes") or [] if self._node_observed_in_recording(node)]
        node_lookup = {node.get("id"): node for node in all_nodes}
        contains_edges = [edge for edge in graph.get("edges") or [] if edge.get("relation") == "contains"]
        contained_ids = {edge.get("dst_id") for edge in contains_edges}
        selected = [
            node
            for node in all_nodes
            if node.get("type") in {"room", "portal", "container"} or node.get("id") in contained_ids
        ]
        slot_rows = {
            "room": (int(panel_height * 0.18), 5),
            "portal": (int(panel_height * 0.40), 7),
            "container": (int(panel_height * 0.62), 6),
            "object": (int(panel_height * 0.78), 8),
        }
        positions = {}
        for node in sorted(selected, key=lambda item: str(item.get("id"))):
            node_id = str(node.get("id"))
            node_type = str(node.get("type", "object"))
            if node_id not in self.topology_slots:
                slot = int(self.topology_next_slot.get(node_type, 0))
                self.topology_next_slot[node_type] = slot + 1
                self.topology_slots[node_id] = (slot, slot_rows.get(node_type, slot_rows["object"])[1])
            slot, columns = self.topology_slots[node_id]
            row_y, _ = slot_rows.get(node_type, slot_rows["object"])
            column = slot % max(1, columns)
            extra_row = slot // max(1, columns)
            x = int((column + 1) * panel_width / (columns + 1))
            y = min(panel_height - 74, row_y + extra_row * 34)
            positions[node_id] = (x, y)
        for edge in graph.get("edges") or []:
            src = positions.get(edge.get("src_id"))
            dst = positions.get(edge.get("dst_id"))
            relation = edge.get("relation")
            if src is None or dst is None or relation not in {"connects", "adjacent_via", "contains"}:
                continue
            color = (220, 85, 45) if relation in {"connects", "adjacent_via"} else (170, 75, 210) if relation == "contains" else (90, 130, 190)
            cv2.line(panel, src, dst, color, 2, cv2.LINE_AA)
        for node in selected:
            center = positions.get(node.get("id"))
            if center is None:
                continue
            color = self._semantic_node_color(node)
            cv2.circle(panel, center, 13 if node.get("type") == "room" else 9, color, -1, cv2.LINE_AA)
            state = str((node.get("interaction") or {}).get("state", ""))
            label = f"{self._short_node_id(node)} {node.get('label', node.get('type', ''))}"
            if state and state != "unknown":
                label += f"[{state}]"
            cv2.putText(panel, label[:18], (center[0] - 24, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (30, 30, 30), 1, cv2.LINE_AA)
        revision = int(graph.get("graph_revision", 0) or 0)
        cv2.putText(panel, f"INTERACTION TOPOLOGY rev={revision}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
        legend = "ROOM -- PORTAL -- ROOM    CONTAINER --contains--> OBJECT"
        cv2.putText(panel, legend, (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (75, 75, 75), 1, cv2.LINE_AA)
        event_y = panel_height - 62
        cv2.rectangle(panel, (6, event_y - 20), (panel_width - 6, panel_height - 6), (238, 238, 238), -1)
        for index, event in enumerate(self.semantic_events[-3:]):
            text = f"r{event.get('graph_revision', 0)} {event.get('event', '')} {event.get('node_id') or event.get('relation', '')}"
            cv2.putText(panel, text[:58], (12, event_y + index * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (45, 45, 45), 1, cv2.LINE_AA)
        return panel

    def _record_first_person_video_frame_locked(self, width: int, height: int, rgb: bytearray) -> None:
        if not self.args.first_person_video or cv2 is None or np is None:
            return
        now = time.time()
        capture_fps = max(0.1, float(self.args.first_person_video_capture_fps))
        if self.last_first_person_video_frame_time > 0.0 and now - self.last_first_person_video_frame_time < 1.0 / capture_fps:
            return
        try:
            with self.lock:
                image_stamp = 0.0 if self.latest_image is None else float(self.latest_image[0])
                image_step = int(self.latest_image_step)
                map_grid = self.latest_grid
                map_base = self.latest_grid_video_rgb
                map_stamp = float(self.latest_grid_video_stamp)
                map_step = int(self.latest_grid_step)
                map_wall_time = float(self.latest_grid_wall_time)
                global_costmap = self.latest_global_costmap
                global_costmap_base = self.latest_global_costmap_video_rgb
                global_costmap_step = int(self.latest_global_costmap_step)
                local_costmap = self.latest_local_costmap
                local_costmap_base = self.latest_local_costmap_video_rgb
                local_costmap_step = int(self.latest_local_costmap_step)
                pose = self.latest_pose
                trajectory = list(self.trajectory)
                active_goal = self._active_goal_xy_locked()
                active_goal_yaw = self._active_goal_yaw_locked()
                global_plan = self.latest_global_plan
                local_global_plan = self.latest_local_global_plan
                local_plan = self.latest_local_plan
                distance_m = float(self.distance_m)
                goal_count = int(self.goal_count)
            stamp_delta = abs(image_stamp - map_stamp) if image_stamp > 0.0 and map_stamp > 0.0 else float("inf")
            map_available = map_grid is not None and map_base is not None
            map_age = now - map_wall_time if map_wall_time > 0.0 else float("inf")
            map_fresh = map_available and map_age <= float(self.args.video_map_max_age_sec)
            map_sync = map_available
            frame_width = int(self.args.first_person_video_width_px)
            frame_height = int(round(height * frame_width / max(width, 1)))
            frame_width = max(1, frame_width)
            frame_height = max(1, frame_height)
            camera_frame = np.frombuffer(bytes(rgb), dtype=np.uint8).reshape((height, width, 3))
            camera_frame = cv2.resize(camera_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            if self.args.semantic_video:
                self._draw_gt_observations_locked(camera_frame, width, height)
            occ_panel = None
            global_costmap_panel = None
            local_costmap_panel = None
            semantic_spatial_panel = None
            semantic_topology_panel = None
            if self.args.first_person_video_with_map:
                occ_panel = self._render_video_map_panel_locked(
                    frame_width,
                    frame_height,
                    grid=map_grid,
                    base=map_base,
                    title=f"OCC step={_step4(map_step)}",
                    pose=pose,
                    goal_xy=active_goal,
                    goal_yaw=active_goal_yaw,
                    trajectory=trajectory,
                    global_plan=global_plan,
                    local_plan=local_plan,
                    image_step=image_step,
                )
                global_costmap_panel = self._render_video_map_panel_locked(
                    frame_width,
                    frame_height,
                    grid=global_costmap,
                    base=global_costmap_base,
                    title=f"GLOBAL step={_step4(global_costmap_step)}",
                    bbox_attr="video_global_costmap_bbox",
                    draw_frontiers=False,
                    draw_global_plan=True,
                    draw_local_global_plan=False,
                    draw_local_plan=False,
                    draw_goal=True,
                    pose=pose,
                    goal_xy=active_goal,
                    goal_yaw=active_goal_yaw,
                    trajectory=trajectory,
                    global_plan=global_plan,
                    image_step=image_step,
                )
                local_costmap_panel = self._render_video_map_panel_locked(
                    frame_width,
                    frame_height,
                    grid=local_costmap,
                    base=local_costmap_base,
                    title=f"LOCAL step={_step4(local_costmap_step)}",
                    bbox_attr="video_local_costmap_bbox",
                    draw_frontiers=False,
                    draw_global_plan=False,
                    draw_local_global_plan=True,
                    draw_local_plan=True,
                    draw_goal=False,
                    pose=pose,
                    goal_xy=active_goal,
                    goal_yaw=active_goal_yaw,
                    trajectory=trajectory,
                    local_global_plan=local_global_plan,
                    local_plan=local_plan,
                    image_step=image_step,
                )
            if self.args.semantic_video:
                semantic_spatial_panel = self._render_semantic_spatial_panel_locked(frame_width, frame_height, pose)
                semantic_topology_panel = self._render_semantic_topology_panel_locked(frame_width, frame_height)
            video_size = (frame_width, frame_height)
            if (
                occ_panel is not None
                and global_costmap_panel is not None
                and local_costmap_panel is not None
                and semantic_spatial_panel is not None
                and semantic_topology_panel is not None
            ):
                video_size = (frame_width * 3, frame_height * 2)
            elif occ_panel is not None and global_costmap_panel is not None and local_costmap_panel is not None:
                video_size = (frame_width * 2, frame_height * 2)
            if self.first_person_video_size is None:
                self.first_person_video_size = video_size
            target_size = self.first_person_video_size or (frame_width, frame_height)
            if (
                occ_panel is not None
                and global_costmap_panel is not None
                and local_costmap_panel is not None
                and semantic_spatial_panel is not None
                and semantic_topology_panel is not None
                and target_size == (frame_width * 3, frame_height * 2)
            ):
                frame = np.vstack(
                    [
                        np.concatenate([camera_frame, occ_panel, semantic_spatial_panel], axis=1),
                        np.concatenate([global_costmap_panel, local_costmap_panel, semantic_topology_panel], axis=1),
                    ]
                )
            elif (
                occ_panel is not None
                and global_costmap_panel is not None
                and local_costmap_panel is not None
                and target_size == (frame_width * 2, frame_height * 2)
            ):
                frame = np.vstack(
                    [
                        np.concatenate([camera_frame, occ_panel], axis=1),
                        np.concatenate([global_costmap_panel, local_costmap_panel], axis=1),
                    ]
                )
            elif target_size != (frame_width, frame_height):
                frame = cv2.resize(camera_frame, target_size, interpolation=cv2.INTER_AREA)
            else:
                frame = camera_frame
            step_gap = abs(int(image_step) - int(map_step))
            desync = f"  MAP_GAP={step_gap}" if step_gap > int(self.args.video_map_desync_step_warn) else ""
            stuck = self._stuck_test_locked(now)
            map_age_text = "inf" if not math.isfinite(map_age) else f"{map_age:.2f}s"
            stamp_delta_text = "inf" if not math.isfinite(stamp_delta) else f"{stamp_delta:.3f}s"
            active_flag = 1 if active_goal is not None else 0
            label = f"STEP={_step4(image_step)} OCC={_step4(map_step)} G={_step4(global_costmap_step)} L={_step4(local_costmap_step)}{desync}"
            label2 = (
                f"dist={distance_m:.2f}m goal=#{goal_count:03d} active={active_flag} "
                f"occ_age={map_age_text} fresh={int(map_fresh)} hdr_dT={stamp_delta_text}"
            )
            stuck_label = (
                f"STUCK_TEST={stuck['state']} dur={stuck['duration_sec']:.1f}s "
                f"move={stuck['moved_m']:.2f}m yaw={stuck['yaw_delta_rad']:.2f}rad"
            )
            text_right = min(frame_width - 8, 620)
            cv2.rectangle(frame, (8, 8), (text_right, 88), (255, 255, 255), -1)
            cv2.putText(frame, label, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (15, 15, 15), 1, cv2.LINE_AA)
            self.first_person_video_frame_count += 1
            frame_index = self.first_person_video_frame_count
            cv2.putText(frame, label2, (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (15, 15, 15), 1, cv2.LINE_AA)
            cv2.putText(frame, stuck_label, (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (15, 15, 15), 1, cv2.LINE_AA)
            camera_path = self.video_camera_frame_dir / f"frame_{frame_index:06d}_camera.png"
            map_path = self.video_map_frame_dir / f"frame_{frame_index:06d}_map.png"
            global_costmap_path = self.video_global_costmap_frame_dir / f"frame_{frame_index:06d}_global_costmap.png"
            local_costmap_path = self.video_local_costmap_frame_dir / f"frame_{frame_index:06d}_local_costmap.png"
            semantic_spatial_path = self.video_semantic_spatial_frame_dir / f"frame_{frame_index:06d}_semantic_spatial.png"
            semantic_topology_path = self.video_semantic_topology_frame_dir / f"frame_{frame_index:06d}_semantic_topology.png"
            composite_path = self.video_composite_frame_dir / f"frame_{frame_index:06d}_composite.png"
            _write_png(camera_path, frame_width, frame_height, bytearray(camera_frame.tobytes()))
            if occ_panel is not None:
                _write_png(map_path, int(occ_panel.shape[1]), int(occ_panel.shape[0]), bytearray(occ_panel.tobytes()))
            if global_costmap_panel is not None:
                _write_png(
                    global_costmap_path,
                    int(global_costmap_panel.shape[1]),
                    int(global_costmap_panel.shape[0]),
                    bytearray(global_costmap_panel.tobytes()),
                )
            if local_costmap_panel is not None:
                _write_png(
                    local_costmap_path,
                    int(local_costmap_panel.shape[1]),
                    int(local_costmap_panel.shape[0]),
                    bytearray(local_costmap_panel.tobytes()),
                )
            if semantic_spatial_panel is not None:
                _write_png(
                    semantic_spatial_path,
                    int(semantic_spatial_panel.shape[1]),
                    int(semantic_spatial_panel.shape[0]),
                    bytearray(semantic_spatial_panel.tobytes()),
                )
            if semantic_topology_panel is not None:
                _write_png(
                    semantic_topology_path,
                    int(semantic_topology_panel.shape[1]),
                    int(semantic_topology_panel.shape[0]),
                    bytearray(semantic_topology_panel.tobytes()),
                )
            _write_png(composite_path, int(frame.shape[1]), int(frame.shape[0]), bytearray(frame.tobytes()))
            graph_revision = int(self.latest_unified_graph.get("graph_revision", 0) or 0)
            if (
                graph_revision == self.pending_semantic_keyframe_revision
                and graph_revision != self.last_semantic_keyframe_revision
            ):
                keyframe_path = self.semantic_keyframe_dir / f"revision_{graph_revision:06d}.png"
                _write_png(keyframe_path, int(frame.shape[1]), int(frame.shape[0]), bytearray(frame.tobytes()))
                self.last_semantic_keyframe_revision = graph_revision
            record = {
                "frame_index": frame_index,
                "step_id": image_step,
                "elapsed_sec": now - self.start_wall_time,
                "image_stamp": image_stamp,
                "map_stamp": map_stamp,
                "stamp_delta_sec": stamp_delta,
                "map_sync": map_sync,
                "map_fresh": map_fresh,
                "map_age_wall_sec": map_age,
                "distance_m": distance_m,
                "goal_count": goal_count,
                "robot_pose": list(pose) if pose is not None else None,
                "active_goal": list(active_goal) if active_goal is not None else None,
                "stuck": stuck,
                "camera_frame": str(camera_path),
                "map_frame": str(map_path) if occ_panel is not None else "",
                "global_costmap_frame": str(global_costmap_path) if global_costmap_panel is not None else "",
                "local_costmap_frame": str(local_costmap_path) if local_costmap_panel is not None else "",
                "semantic_spatial_frame": str(semantic_spatial_path) if semantic_spatial_panel is not None else "",
                "semantic_topology_frame": str(semantic_topology_path) if semantic_topology_panel is not None else "",
                "graph_revision": graph_revision,
                "composite_frame": str(composite_path),
            }
            self.first_person_video_frames.append(record)
            self.video_frames_writer.writerow(
                {
                    "frame_index": frame_index,
                    "step_id": image_step,
                    "elapsed_sec": f"{record['elapsed_sec']:.3f}",
                    "image_stamp": f"{image_stamp:.6f}" if image_stamp > 0.0 else "",
                    "map_stamp": f"{map_stamp:.6f}" if map_stamp > 0.0 else "",
                    "stamp_delta_sec": "" if not math.isfinite(stamp_delta) else f"{stamp_delta:.6f}",
                    "map_sync": map_sync,
                    "map_fresh": map_fresh,
                    "map_age_wall_sec": "" if not math.isfinite(map_age) else f"{map_age:.6f}",
                    "distance_m": f"{distance_m:.6f}",
                    "goal_count": goal_count,
                    "robot_x": "" if pose is None else f"{pose[0]:.6f}",
                    "robot_y": "" if pose is None else f"{pose[1]:.6f}",
                    "robot_yaw": "" if pose is None else f"{pose[2]:.6f}",
                    "active_goal": "" if active_goal is None else f"{active_goal[0]:.6f},{active_goal[1]:.6f}",
                    "stuck_state": stuck["state"],
                    "stuck_duration_sec": f"{stuck['duration_sec']:.3f}",
                    "stuck_moved_m": f"{stuck['moved_m']:.6f}",
                    "stuck_yaw_delta_rad": f"{stuck['yaw_delta_rad']:.6f}",
                    "camera_frame": str(camera_path),
                    "map_frame": str(map_path) if occ_panel is not None else "",
                    "global_costmap_step": global_costmap_step,
                    "local_costmap_step": local_costmap_step,
                    "global_costmap_frame": str(global_costmap_path) if global_costmap_panel is not None else "",
                    "local_costmap_frame": str(local_costmap_path) if local_costmap_panel is not None else "",
                    "semantic_spatial_frame": str(semantic_spatial_path) if semantic_spatial_panel is not None else "",
                    "semantic_topology_frame": str(semantic_topology_path) if semantic_topology_panel is not None else "",
                    "composite_frame": str(composite_path),
                }
            )
            self.last_first_person_video_frame_time = now
        except Exception as exc:  # pragma: no cover - debug recorder should not crash ROS
            self.first_person_video_error = f"{type(exc).__name__}: {exc}"

    def _record_external_video_frame_locked(self, width: int, height: int, rgb: bytearray, image_stamp: float) -> None:
        if not self.args.external_video or not self.args.external_image_topic or cv2 is None or np is None:
            return
        now = time.time()
        fps = max(0.1, float(self.args.first_person_video_fps))
        if self.last_external_video_frame_time > 0.0 and now - self.last_external_video_frame_time < 1.0 / fps:
            return
        try:
            frame_width = int(self.args.first_person_video_width_px)
            frame_height = int(round(height * frame_width / max(width, 1)))
            frame_width = max(1, frame_width)
            frame_height = max(1, frame_height)
            frame = np.frombuffer(bytes(rgb), dtype=np.uint8).reshape((height, width, 3))
            frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            stuck = self._stuck_test_locked(now)
            active_goal = self._active_goal_xy_locked()
            active_flag = 1 if active_goal is not None else 0
            label = (
                f"EXT STEP={_step4(self.latest_external_image_step)} IMG_STAMP={image_stamp:.3f} "
                f"dist={self.distance_m:.2f}m last_goal=#{self.goal_count:03d} active={active_flag}"
            )
            stuck_label = (
                f"STUCK_TEST={stuck['state']} dur={stuck['duration_sec']:.1f}s "
                f"move={stuck['moved_m']:.2f}m yaw={stuck['yaw_delta_rad']:.2f}rad"
            )
            cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 8, 900), 66), (255, 255, 255), -1)
            cv2.putText(frame, label, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 15, 15), 2, cv2.LINE_AA)
            cv2.putText(frame, stuck_label, (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (15, 15, 15), 2, cv2.LINE_AA)
            self.external_video_frame_count += 1
            frame_index = self.external_video_frame_count
            frame_path = self.video_external_frame_dir / f"frame_{frame_index:06d}_external.png"
            _write_png(frame_path, int(frame.shape[1]), int(frame.shape[0]), bytearray(frame.tobytes()))
            self.external_video_frames.append(
                {
                    "frame_index": frame_index,
                    "step_id": self.latest_external_image_step,
                    "elapsed_sec": now - self.start_wall_time,
                    "image_stamp": image_stamp,
                    "distance_m": self.distance_m,
                    "goal_count": self.goal_count,
                    "active_goal": list(active_goal) if active_goal is not None else None,
                    "stuck": stuck,
                    "frame": str(frame_path),
                }
            )
            self.last_external_video_frame_time = now
        except Exception as exc:  # pragma: no cover - debug recorder should not crash ROS
            self.external_video_error = f"{type(exc).__name__}: {exc}"

    def _render_unsynced_map_panel(self, panel_width: int, panel_height: int, image_stamp: float, map_stamp: float, delta: float):
        if cv2 is None or np is None:
            return None
        panel = np.full((panel_height, panel_width, 3), 235, dtype=np.uint8)
        delta_text = "inf" if not math.isfinite(delta) else f"{delta:.3f}s"
        lines = [
            "NO STRICT SYNC MAP",
            f"image_stamp={image_stamp:.6f}" if image_stamp > 0.0 else "image_stamp=<none>",
            f"map_stamp={map_stamp:.6f}" if map_stamp > 0.0 else "map_stamp=<none>",
            f"delta={delta_text} max={float(self.args.video_sync_max_delta_sec):.3f}s",
        ]
        y = 38
        for line in lines:
            cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (70, 70, 70), 2, cv2.LINE_AA)
            y += 34
        return panel

    def _stuck_test_locked(self, now: float) -> dict:
        active_goal = self._active_goal_xy_locked()
        pose = self.latest_pose
        if active_goal is None:
            return {
                "state": "NO_ACTIVE_GOAL",
                "duration_sec": 0.0,
                "moved_m": 0.0,
                "yaw_delta_rad": 0.0,
            }
        if pose is None or self.stall_reference_xy is None:
            return {
                "state": "NO_POSE",
                "duration_sec": 0.0,
                "moved_m": 0.0,
                "yaw_delta_rad": 0.0,
            }
        duration = max(0.0, now - self.stall_reference_time)
        moved = math.hypot(pose[0] - self.stall_reference_xy[0], pose[1] - self.stall_reference_xy[1])
        yaw_delta = 0.0
        if self.stall_reference_yaw is not None:
            yaw_delta = self._angle_distance(pose[2], self.stall_reference_yaw)
        if duration < float(self.args.video_stuck_window_sec):
            state = "OBSERVING"
        elif moved > float(self.args.video_stuck_distance_m):
            state = "MOVING"
        elif yaw_delta >= float(self.args.video_stuck_rotation_yaw_rad):
            state = "STUCK_ROTATING"
        else:
            state = "STUCK_STATIC"
        return {
            "state": state,
            "duration_sec": duration,
            "moved_m": moved,
            "yaw_delta_rad": yaw_delta,
        }

    @staticmethod
    def _angle_distance(a: float, b: float) -> float:
        return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)

    def _video_writer_path(self) -> str:
        if self.args.first_person_video_h264:
            return self.first_person_video_raw_path
        return self.first_person_video_path

    def _finalize_first_person_video_locked(self) -> None:
        if self.first_person_video_writer is not None:
            self.first_person_video_writer.release()
            self.first_person_video_writer = None
        if self.args.first_person_video and self.first_person_video_frame_count > 0:
            self._build_first_person_video_from_frames_locked()
        if not self.args.first_person_video_h264 or self.first_person_video_frame_count <= 0:
            return
        raw_path = Path(self.first_person_video_raw_path)
        final_path = Path(self.first_person_video_path)
        if not raw_path.exists() or raw_path.stat().st_size <= 0:
            self.first_person_video_error = self.first_person_video_error or "raw_video_missing"
            return
        temp_path = final_path.with_name(f"{final_path.stem}_h264_tmp{final_path.suffix}")
        log_path = final_path.with_name(f"{final_path.stem}_h264_ffmpeg.log")
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(raw_path),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(self.args.first_person_video_h264_crf),
            "-preset",
            str(self.args.first_person_video_h264_preset),
            str(temp_path),
        ]
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=max(1.0, float(self.args.first_person_video_h264_timeout_sec)),
                )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - best-effort debug artifact
            self.first_person_video_error = f"h264_transcode_timeout:{type(exc).__name__}: {exc}"
            try:
                temp_path.unlink()
            except OSError:
                pass
            return
        except Exception as exc:  # pragma: no cover - best-effort debug artifact
            self.first_person_video_error = f"h264_transcode_exception:{type(exc).__name__}: {exc}"
            try:
                temp_path.unlink()
            except OSError:
                pass
            return
        if completed.returncode != 0 or not temp_path.exists() or temp_path.stat().st_size <= 0:
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
            except OSError:
                tail = []
            self.first_person_video_error = "h264_transcode_failed:" + " | ".join(tail)
            try:
                temp_path.unlink()
            except OSError:
                pass
            return
        temp_path.replace(final_path)
        self.first_person_video_codec_name = "h264"
        try:
            raw_path.unlink()
        except OSError as exc:
            suffix = f"raw_video_cleanup_failed:{type(exc).__name__}: {exc}"
            self.first_person_video_error = suffix if not self.first_person_video_error else f"{self.first_person_video_error};{suffix}"

    def _build_first_person_video_from_frames_locked(self) -> None:
        if cv2 is None or np is None:
            self.first_person_video_error = self.first_person_video_error or "cv2_or_numpy_unavailable"
            return
        output_path = Path(self._video_writer_path())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_records = list(self.first_person_video_frames)
        if not frame_records:
            self.first_person_video_error = self.first_person_video_error or "no_video_frames"
            return
        first = _read_png(Path(frame_records[0].get("composite_frame", "")))
        if first is None:
            self.first_person_video_error = self.first_person_video_error or "first_video_frame_missing"
            return
        width, height, _rgb = first
        fourcc = cv2.VideoWriter_fourcc(*str(self.args.first_person_video_codec)[:4])
        writer = cv2.VideoWriter(str(output_path), fourcc, max(0.1, float(self.args.first_person_video_fps)), (width, height))
        if not writer.isOpened():
            self.first_person_video_error = self.first_person_video_error or "video_writer_open_failed"
            return
        written = 0
        try:
            for record in frame_records:
                loaded = _read_png(Path(record.get("composite_frame", "")))
                if loaded is None:
                    continue
                frame_w, frame_h, frame_rgb = loaded
                frame = np.frombuffer(bytes(frame_rgb), dtype=np.uint8).reshape((frame_h, frame_w, 3))
                if frame_w != width or frame_h != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                written += 1
        finally:
            writer.release()
        if written <= 0 or not output_path.exists() or output_path.stat().st_size <= 0:
            self.first_person_video_error = self.first_person_video_error or "video_writer_no_frames"
        self.first_person_video_frame_count = written

    def _finalize_external_video_locked(self) -> None:
        if not self.args.external_video or self.external_video_frame_count <= 0:
            return
        self._build_external_video_from_frames_locked()
        if not self.args.first_person_video_h264 or self.external_video_frame_count <= 0:
            return
        raw_path = Path(self.external_video_raw_path)
        final_path = Path(self.external_video_path)
        if not raw_path.exists() or raw_path.stat().st_size <= 0:
            self.external_video_error = self.external_video_error or "raw_video_missing"
            return
        temp_path = final_path.with_name(f"{final_path.stem}_h264_tmp{final_path.suffix}")
        log_path = final_path.with_name(f"{final_path.stem}_h264_ffmpeg.log")
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(raw_path),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(self.args.first_person_video_h264_crf),
            "-preset",
            str(self.args.first_person_video_h264_preset),
            str(temp_path),
        ]
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=max(1.0, float(self.args.first_person_video_h264_timeout_sec)),
                )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - best-effort debug artifact
            self.external_video_error = f"h264_transcode_timeout:{type(exc).__name__}: {exc}"
            try:
                temp_path.unlink()
            except OSError:
                pass
            return
        except Exception as exc:  # pragma: no cover - best-effort debug artifact
            self.external_video_error = f"h264_transcode_exception:{type(exc).__name__}: {exc}"
            try:
                temp_path.unlink()
            except OSError:
                pass
            return
        if completed.returncode != 0 or not temp_path.exists() or temp_path.stat().st_size <= 0:
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
            except OSError:
                tail = []
            self.external_video_error = "h264_transcode_failed:" + " | ".join(tail)
            try:
                temp_path.unlink()
            except OSError:
                pass
            return
        temp_path.replace(final_path)
        self.external_video_codec_name = "h264"
        try:
            raw_path.unlink()
        except OSError as exc:
            suffix = f"raw_video_cleanup_failed:{type(exc).__name__}: {exc}"
            self.external_video_error = suffix if not self.external_video_error else f"{self.external_video_error};{suffix}"

    def _build_external_video_from_frames_locked(self) -> None:
        if cv2 is None or np is None:
            self.external_video_error = self.external_video_error or "cv2_or_numpy_unavailable"
            return
        output_path = Path(self.external_video_raw_path if self.args.first_person_video_h264 else self.external_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_records = list(self.external_video_frames)
        if not frame_records:
            self.external_video_error = self.external_video_error or "no_video_frames"
            return
        first = _read_png(Path(frame_records[0].get("frame", "")))
        if first is None:
            self.external_video_error = self.external_video_error or "first_video_frame_missing"
            return
        width, height, _rgb = first
        fourcc = cv2.VideoWriter_fourcc(*str(self.args.first_person_video_codec)[:4])
        writer = cv2.VideoWriter(str(output_path), fourcc, max(0.1, float(self.args.first_person_video_fps)), (width, height))
        if not writer.isOpened():
            self.external_video_error = self.external_video_error or "video_writer_open_failed"
            return
        written = 0
        try:
            for record in frame_records:
                loaded = _read_png(Path(record.get("frame", "")))
                if loaded is None:
                    continue
                frame_w, frame_h, frame_rgb = loaded
                frame = np.frombuffer(bytes(frame_rgb), dtype=np.uint8).reshape((frame_h, frame_w, 3))
                if frame_w != width or frame_h != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                written += 1
        finally:
            writer.release()
        if written <= 0 or not output_path.exists() or output_path.stat().st_size <= 0:
            self.external_video_error = self.external_video_error or "video_writer_no_frames"
        self.external_video_frame_count = written

    def _grid_to_video_rgb_locked(self, grid: OccupancyGrid):
        if np is None:
            return None
        width = int(grid.info.width)
        height = int(grid.info.height)
        if width <= 0 or height <= 0:
            return None
        raw_values = np.asarray(grid.data, dtype=np.int16).reshape((height, width))
        free = (raw_values >= 0) & (raw_values <= 20)
        unknown = raw_values < 0
        unknown_neighbor = np.zeros_like(unknown, dtype=bool)
        unknown_neighbor[:, 1:] |= unknown[:, :-1]
        unknown_neighbor[:, :-1] |= unknown[:, 1:]
        unknown_neighbor[1:, :] |= unknown[:-1, :]
        unknown_neighbor[:-1, :] |= unknown[1:, :]
        raw_frontier = free & unknown_neighbor
        values = np.flipud(raw_values)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[values < 0] = (178, 178, 178)
        rgb[(values >= 0) & (values <= 20)] = (248, 248, 245)
        rgb[(values > 20) & (values < 50)] = (118, 118, 118)
        rgb[values >= 50] = (28, 30, 32)
        rgb[np.flipud(raw_frontier)] = (112, 36, 170)
        return rgb

    def _render_video_map_panel_locked(
        self,
        panel_width: int,
        panel_height: int,
        grid: OccupancyGrid | None = None,
        base=None,
        title: str | None = None,
        bbox_attr: str = "video_map_bbox",
        draw_frontiers: bool = True,
        draw_global_plan: bool = True,
        draw_local_global_plan: bool = False,
        draw_local_plan: bool = True,
        draw_goal: bool = True,
        pose: tuple[float, float, float] | None = None,
        goal_xy: tuple[float, float] | None = None,
        goal_yaw: float | None = None,
        trajectory: list[tuple[float, float, float, float]] | None = None,
        global_plan: dict | None = None,
        local_global_plan: dict | None = None,
        local_plan: dict | None = None,
        image_step: int | None = None,
    ):
        if cv2 is None or np is None:
            return None
        if grid is None:
            grid = self.latest_grid
        if base is None and grid is self.latest_grid:
            base = self.latest_grid_video_rgb
        if title is None:
            title = f"OCC MAP step={_step4(self.latest_grid_step)}"
        if grid is None or base is None:
            panel = np.full((panel_height, panel_width, 3), 235, dtype=np.uint8)
            cv2.putText(panel, f"NO {title} YET", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2, cv2.LINE_AA)
            return panel
        height, width = base.shape[:2]
        if pose is None:
            pose = self.latest_pose
        if goal_xy is None:
            goal_xy = self._active_goal_xy_locked()
        if goal_yaw is None:
            goal_yaw = self._active_goal_yaw_locked()
        if trajectory is None:
            trajectory = self.trajectory
        if global_plan is None:
            global_plan = self.latest_global_plan
        if local_global_plan is None:
            local_global_plan = self.latest_local_global_plan
        if local_plan is None:
            local_plan = self.latest_local_plan
        if image_step is None:
            image_step = self.latest_image_step
        grid_frame = grid.header.frame_id or ""
        if draw_global_plan and goal_xy is not None and not self._plan_reaches_goal(global_plan, goal_xy):
            global_plan = None
        pose_in_grid = self._pose_to_grid_frame(grid, pose, self.args.odom_frame)
        goal_in_grid = self._point_to_grid_frame(grid, goal_xy, self.args.map_frame, 0.0 if goal_yaw is None else goal_yaw)
        global_plan_poses = self._plan_poses_for_grid(grid, global_plan) if draw_global_plan else None
        local_global_plan_poses = self._plan_poses_for_grid(grid, local_global_plan) if draw_local_global_plan else None
        local_plan_poses = self._plan_poses_for_grid(grid, local_plan) if draw_local_plan else None
        has_active_goal = draw_goal and goal_xy is not None

        def cell_to_px(cell: tuple[int, int] | None) -> tuple[int, int] | None:
            if cell is None:
                return None
            return int(cell[0]), int(height - 1 - cell[1])

        def world_to_px(x: float, y: float) -> tuple[int, int] | None:
            return cell_to_px(_world_to_cell(grid, x, y))

        points: list[tuple[int, int]] = []
        for _, x, y, _ in trajectory:
            transformed = self._transform_xy_yaw_to_frame(x, y, 0.0, self.args.odom_frame, grid_frame)
            if transformed is None:
                continue
            px = world_to_px(transformed[0], transformed[1])
            if px is not None:
                points.append(px)
        if pose_in_grid is not None:
            px = world_to_px(pose_in_grid[0], pose_in_grid[1])
            if px is not None:
                points.append(px)
        if goal_in_grid is not None:
            px = world_to_px(goal_in_grid[0], goal_in_grid[1])
            if px is not None:
                points.append(px)
        if has_active_goal or draw_global_plan or draw_local_plan:
            plans = []
            if draw_global_plan:
                plans.append(global_plan_poses)
            if draw_local_global_plan:
                plans.append(local_global_plan_poses)
            if draw_local_plan:
                plans.append(local_plan_poses)
            for plan in plans:
                if not plan:
                    continue
                for pose in plan:
                    px = world_to_px(float(pose[0]), float(pose[1]))
                    if px is not None:
                        points.append(px)
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            margin = max(40, int(self.args.video_map_crop_margin_px))
            current_bbox = (
                max(0, min(xs) - margin),
                max(0, min(ys) - margin),
                min(width - 1, max(xs) + margin),
                min(height - 1, max(ys) + margin),
            )
            old_bbox = getattr(self, bbox_attr, None)
            if old_bbox is None:
                setattr(self, bbox_attr, current_bbox)
            else:
                old = old_bbox
                setattr(
                    self,
                    bbox_attr,
                    (
                    max(0, min(old[0], current_bbox[0])),
                    max(0, min(old[1], current_bbox[1])),
                    min(width - 1, max(old[2], current_bbox[2])),
                    min(height - 1, max(old[3], current_bbox[3])),
                    ),
                )
            min_x, min_y, max_x, max_y = getattr(self, bbox_attr)
        else:
            min_x, min_y, max_x, max_y = 0, 0, width - 1, height - 1
        if max_x <= min_x or max_y <= min_y:
            return None
        crop = base[min_y : max_y + 1, min_x : max_x + 1].copy()
        crop_h, crop_w = crop.shape[:2]
        scale = min(float(panel_width) / max(crop_w, 1), float(panel_height) / max(crop_h, 1))
        scaled_w = max(1, int(round(crop_w * scale)))
        scaled_h = max(1, int(round(crop_h * scale)))
        resized = cv2.resize(crop, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
        panel = np.full((panel_height, panel_width, 3), 235, dtype=np.uint8)
        offset_x = (panel_width - scaled_w) // 2
        offset_y = (panel_height - scaled_h) // 2
        panel[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w] = resized

        def to_panel(px: tuple[int, int] | None) -> tuple[int, int] | None:
            if px is None:
                return None
            x, y = px
            if x < min_x or x > max_x or y < min_y or y > max_y:
                return None
            return int(round(offset_x + (x - min_x) * scale)), int(round(offset_y + (y - min_y) * scale))

        frontier_points: list[tuple[int, int]] = []
        if draw_frontiers and goal_in_grid is not None:
            goal_cell = _world_to_cell(grid, goal_in_grid[0], goal_in_grid[1])
            if goal_cell is not None:
                radius_cells = max(1, int(math.ceil(self.args.frontier_check_radius_m / max(float(grid.info.resolution), 1e-6))))
                gx, gy = goal_cell
                for cy in range(gy - radius_cells, gy + radius_cells + 1):
                    if cy < 0 or cy >= int(grid.info.height):
                        continue
                    for cx in range(gx - radius_cells, gx + radius_cells + 1):
                        if cx < 0 or cx >= int(grid.info.width):
                            continue
                        if not _is_free(int(grid.data[cy * int(grid.info.width) + cx])):
                            continue
                        has_unknown_neighbor = False
                        for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                            if nx < 0 or ny < 0 or nx >= int(grid.info.width) or ny >= int(grid.info.height):
                                continue
                            if _is_unknown(int(grid.data[ny * int(grid.info.width) + nx])):
                                has_unknown_neighbor = True
                                break
                        if has_unknown_neighbor:
                            panel_px = to_panel(cell_to_px((cx, cy)))
                            if panel_px is not None:
                                frontier_points.append(panel_px)
        frontier_radius = max(1, int(round(2.0 * max(scale, 1.0))))
        for frontier_px in frontier_points:
            cv2.circle(panel, frontier_px, frontier_radius, (112, 36, 170), -1, cv2.LINE_AA)

        trajectory_points = []
        for _, x, y, _ in trajectory:
            transformed = self._transform_xy_yaw_to_frame(x, y, 0.0, self.args.odom_frame, grid_frame)
            if transformed is not None:
                trajectory_points.append(to_panel(world_to_px(transformed[0], transformed[1])))
        trajectory_points = [p for p in trajectory_points if p is not None]
        self._draw_cv_polyline(panel, trajectory_points, (20, 118, 230), 3)
        if pose_in_grid is not None:
            robot_px = to_panel(world_to_px(pose_in_grid[0], pose_in_grid[1]))
            if robot_px is not None:
                self._draw_cv_robot_arrow(panel, robot_px, pose_in_grid[2], max(9, int(9 * scale)))
        if draw_global_plan and global_plan_poses:
            plan_points = [to_panel(world_to_px(float(p[0]), float(p[1]))) for p in global_plan_poses]
            self._draw_cv_polyline(panel, [p for p in plan_points if p is not None], (40, 190, 60), 3)
        if draw_local_global_plan and local_global_plan_poses:
            plan_points = [to_panel(world_to_px(float(p[0]), float(p[1]))) for p in local_global_plan_poses]
            self._draw_cv_polyline(panel, [p for p in plan_points if p is not None], (40, 190, 60), 3)
        if draw_local_plan and local_plan_poses:
            plan_points = [to_panel(world_to_px(float(p[0]), float(p[1]))) for p in local_plan_poses]
            self._draw_cv_polyline(panel, [p for p in plan_points if p is not None], (240, 150, 20), 3)
        if goal_in_grid is not None:
            goal_px = to_panel(world_to_px(goal_in_grid[0], goal_in_grid[1]))
            if goal_px is not None:
                self._draw_cv_goal_arrow(panel, goal_px, goal_in_grid[2], max(9, int(9 * scale)))
        cv2.rectangle(panel, (0, 0), (min(panel.shape[1] - 1, 680), 32), (255, 255, 255), -1)
        cv2.putText(
            panel,
            f"{title}  IMG STEP={_step4(image_step)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50 if panel_width < 700 else 0.65,
            (20, 20, 20),
            1 if panel_width < 700 else 2,
            cv2.LINE_AA,
        )
        return panel

    @staticmethod
    def _draw_cv_polyline(image, points: list[tuple[int, int]], color: tuple[int, int, int], thickness: int) -> None:
        if len(points) < 2 or cv2 is None or np is None:
            return
        cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_cv_robot_arrow(image, center: tuple[int, int], yaw: float, length: int) -> None:
        if cv2 is None or np is None:
            return
        cx, cy = center
        heading = np.asarray([math.cos(yaw), -math.sin(yaw)], dtype=np.float32)
        norm = float(np.linalg.norm(heading))
        if norm <= 1e-6:
            heading = np.asarray([1.0, 0.0], dtype=np.float32)
        else:
            heading = heading / norm
        perp = np.asarray([-heading[1], heading[0]], dtype=np.float32)
        tip = np.asarray([cx, cy], dtype=np.float32) + heading * float(length)
        back = np.asarray([cx, cy], dtype=np.float32) - heading * float(length * 0.55)
        left = back + perp * float(length * 0.45)
        right = back - perp * float(length * 0.45)
        pts = np.asarray([tip, left, right], dtype=np.int32)
        cv2.fillConvexPoly(image, pts, (0, 88, 255), cv2.LINE_AA)
        cv2.polylines(image, [pts], True, (0, 35, 160), 2, cv2.LINE_AA)

    @staticmethod
    def _draw_cv_goal_arrow(image, center: tuple[int, int], yaw: float, length: int) -> None:
        if cv2 is None or np is None:
            return
        cx, cy = center
        heading = np.asarray([math.cos(yaw), -math.sin(yaw)], dtype=np.float32)
        norm = float(np.linalg.norm(heading))
        if norm <= 1e-6:
            heading = np.asarray([1.0, 0.0], dtype=np.float32)
        else:
            heading = heading / norm
        start = np.asarray([cx, cy], dtype=np.float32) - heading * float(length * 0.45)
        end = np.asarray([cx, cy], dtype=np.float32) + heading * float(length)
        cv2.arrowedLine(
            image,
            tuple(start.astype(np.int32)),
            tuple(end.astype(np.int32)),
            (230, 30, 45),
            4,
            cv2.LINE_AA,
            tipLength=0.45,
        )
        cv2.circle(image, (cx, cy), max(4, length // 4), (230, 30, 45), -1, cv2.LINE_AA)

    def odom_callback(self, msg: Odometry) -> None:
        if self.shutting_down:
            return
        x, y, yaw = _pose_xy_yaw(msg)
        now = time.time()
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
        with self.lock:
            if self.shutting_down:
                return
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
            self.latest_pose_step = self.debug_step
            self.last_odom_xy = (x, y)
            if self.stall_reference_xy is None:
                self.stall_reference_xy = (x, y)
                self.stall_reference_yaw = yaw
                self.stall_reference_time = now
            elif math.hypot(x - self.stall_reference_xy[0], y - self.stall_reference_xy[1]) >= self.args.stall_snapshot_distance_m:
                self.stall_reference_xy = (x, y)
                self.stall_reference_yaw = yaw
                self.stall_reference_time = now
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
                        "step_id": self.debug_step,
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
        goal_yaw = _yaw_from_quaternion(msg.pose.orientation)
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
        with self.lock:
            if self.shutting_down:
                return
            self.goal_count += 1
            goal_index = self.goal_count
            grid_snapshot = self.latest_grid
            step_id = self.debug_step
            grid_step = self.latest_grid_step
            image_step = self.latest_image_step
            pose = self.latest_pose
            image_snapshot = self.latest_image
            external_snapshot = self.latest_external_image
            trajectory = list(self.trajectory)
            global_plan = None
            local_plan = None
            active_xy = self._active_goal_xy_locked()
            active_yaw = self._active_goal_yaw_locked()
            if active_xy is not None and active_yaw is not None:
                if math.hypot(active_xy[0] - goal_xy[0], active_xy[1] - goal_xy[1]) <= 0.25:
                    goal_yaw = active_yaw
            elapsed = time.time() - self.start_wall_time

        analysis = self._analyze_goal(grid_snapshot, pose, goal_xy)
        overlay_path = ""
        overlay_crop_path = ""
        first_person_path = ""
        external_path = ""
        first_person_stamp = 0.0
        panel_path = ""
        if grid_snapshot is not None:
            overlay_path = str(self.overlay_dir / f"subgoal_{goal_index:04d}.png")
            overlay_crop_path = self._render_overlay(
                Path(overlay_path),
                grid_snapshot,
                pose,
                goal_xy,
                trajectory,
                global_plan=global_plan,
                local_plan=local_plan,
                goal_yaw=goal_yaw,
                label_lines=[
                    f"#{goal_index:03d} STEP={_step4(step_id)} MAP={_step4(grid_step)} IMG={_step4(image_step)}",
                    "GOAL="
                    f"({goal_xy[0]:.2f},{goal_xy[1]:.2f}) yaw={goal_yaw:.2f} "
                    f"G={0 if global_plan is None else len(global_plan.get('poses', []))} "
                    f"L={0 if local_plan is None else len(local_plan.get('poses', []))}",
                ],
            )
        if image_snapshot is not None:
            first_person_stamp, image_width, image_height, image_rgb = image_snapshot
            first_person_path = str(self.first_person_dir / f"subgoal_{goal_index:04d}_first_person.png")
            _write_png(Path(first_person_path), image_width, image_height, image_rgb)
        if external_snapshot is not None:
            _external_stamp, external_width, external_height, external_rgb = external_snapshot
            external_path = str(self.external_dir / f"subgoal_{goal_index:04d}_external.png")
            _write_png(Path(external_path), external_width, external_height, external_rgb)
        global_plan_stamp = 0.0 if global_plan is None else float(global_plan.get("stamp", 0.0))
        local_plan_stamp = 0.0 if local_plan is None else float(local_plan.get("stamp", 0.0))
        global_plan_points = 0 if global_plan is None else len(global_plan.get("poses", []))
        local_plan_points = 0 if local_plan is None else len(local_plan.get("poses", []))
        panel_path = self._render_subgoal_panel(
            goal_index,
            elapsed,
            stamp,
            first_person_stamp,
            overlay_crop_path or overlay_path,
            first_person_path,
            step_id=step_id,
            grid_step=grid_step,
            image_step=image_step,
        )
        record = {
            "index": goal_index,
            "step_id": step_id,
            "grid_step": grid_step,
            "image_step": image_step,
            "grid_snapshot": grid_snapshot,
            "trajectory_snapshot": trajectory,
            "elapsed_sec": elapsed,
            "stamp": stamp,
            "goal": list(goal_xy),
            "goal_yaw": goal_yaw,
            "robot_pose": list(pose) if pose is not None else None,
            "analysis": analysis,
            "overlay": overlay_path,
            "overlay_crop": overlay_crop_path,
            "first_person": first_person_path,
            "external_image": external_path,
            "first_person_stamp": first_person_stamp,
            "panel": panel_path,
            "global_plan_points": global_plan_points,
            "global_plan_stamp": global_plan_stamp,
            "local_plan_points": local_plan_points,
            "local_plan_stamp": local_plan_stamp,
        }

        with self.lock:
            if self.shutting_down:
                return
            self.subgoal_records.append(record)
            self.subgoals_writer.writerow(
                {
                    "index": goal_index,
                    "step_id": step_id,
                    "grid_step": grid_step,
                    "image_step": image_step,
                    "elapsed_sec": f"{elapsed:.3f}",
                    "stamp": f"{stamp:.6f}",
                    "x": f"{goal_xy[0]:.6f}",
                    "y": f"{goal_xy[1]:.6f}",
                    "yaw": f"{goal_yaw:.6f}",
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
                    "external_image": external_path,
                    "first_person_stamp": "" if first_person_stamp <= 0.0 else f"{first_person_stamp:.6f}",
                    "panel": panel_path,
                    "global_plan_points": global_plan_points,
                    "global_plan_stamp": "" if global_plan_stamp <= 0.0 else f"{global_plan_stamp:.6f}",
                    "local_plan_points": local_plan_points,
                    "local_plan_stamp": "" if local_plan_stamp <= 0.0 else f"{local_plan_stamp:.6f}",
                }
            )
            self._write_event("subgoal_published", self._json_safe_record(record))

    def current_subgoal_callback(self, msg: PointStamped) -> None:
        if self.shutting_down:
            return
        with self.lock:
            if self.shutting_down:
                return
            self.current_subgoal_count += 1
            self._write_event(
                "current_subgoal",
                {
                    "index": self.current_subgoal_count,
                    "step_id": self.debug_step,
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
            if self.shutting_down:
                return
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
                    "step_id": self.debug_step,
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
                        "step_id": self.debug_step,
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
            if self.shutting_down:
                return
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
                        "step_id": self.debug_step,
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
        for pose_index, pose in enumerate(msg.poses):
            x = float(pose.pose.position.x)
            y = float(pose.pose.position.y)
            yaw = _yaw_from_quaternion(pose.pose.orientation)
            poses.append((x, y, yaw))
        with self.lock:
            if self.shutting_down:
                return
            self.plan_message_counts[plan_type] = self.plan_message_counts.get(plan_type, 0) + 1
            message_index = self.plan_message_counts[plan_type]
            step_id = self.debug_step
            pose_count = len(msg.poses)
            frame_id = msg.header.frame_id or ""
            for pose_index, (x, y, yaw) in enumerate(poses):
                self.plan_writer.writerow(
                    {
                        "step_id": step_id,
                        "elapsed_sec": f"{elapsed:.3f}",
                        "topic": topic,
                        "plan_type": plan_type,
                        "message_index": message_index,
                        "stamp": f"{stamp:.6f}",
                        "frame_id": frame_id,
                        "pose_index": pose_index,
                        "pose_count": pose_count,
                        "x": f"{x:.6f}",
                        "y": f"{y:.6f}",
                        "yaw": f"{yaw:.6f}",
                    }
                )
            snapshot = {
                "topic": topic,
                "step_id": step_id,
                "elapsed_sec": elapsed,
                "stamp": stamp,
                "frame_id": frame_id,
                "poses": poses,
                "message_index": message_index,
            }
            if plan_type == "global":
                self.latest_global_plan = snapshot
            elif plan_type == "local_global":
                self.latest_local_global_plan = snapshot
            else:
                self.latest_local_plan = snapshot
            self.plan_records.setdefault(plan_type, []).append(snapshot)
            self._write_event(
                "plan_update",
                {
                    "elapsed_sec": elapsed,
                    "plan_type": plan_type,
                    "topic": topic,
                    "stamp": stamp,
                    "pose_count": pose_count,
                    "message_index": message_index,
                },
            )

    def _refresh_latest_subgoal_overlay_locked(self) -> None:
        if not self.subgoal_records:
            return
        record = self.subgoal_records[-1]
        if not isinstance(record, dict):
            return
        grid = record.get("grid_snapshot") or self.latest_grid
        if grid is None:
            return
        goal_elapsed = float(record.get("elapsed_sec", 0.0))
        max_pre = float(self.args.plan_match_pre_goal_sec)
        max_post = float(self.args.plan_match_post_goal_sec)
        goal = record.get("goal")
        if not goal or len(goal) < 2:
            return
        goal_xy = (float(goal[0]), float(goal[1]))
        global_plan = self.latest_global_plan
        local_plan = self.latest_local_plan
        if not self._plan_matches_subgoal_window(
            global_plan,
            goal_elapsed,
            max_pre,
            max_post,
            goal_xy=goal_xy,
            require_goal_endpoint=True,
        ):
            global_plan = None
        if not self._plan_matches_subgoal_window(local_plan, goal_elapsed, max_pre, max_post):
            local_plan = None
        have_global = bool(global_plan and global_plan.get("poses"))
        have_local = bool(local_plan and local_plan.get("poses"))
        if not have_global and not have_local:
            return
        current_global = int(record.get("global_plan_points") or 0)
        current_local = int(record.get("local_plan_points") or 0)
        if current_global > 0 and current_local > 0:
            return
        robot_pose = record.get("robot_pose")
        pose = None
        if isinstance(robot_pose, list) and len(robot_pose) >= 3:
            pose = (float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2]))
        overlay_path = record.get("overlay")
        if overlay_path:
            label_lines = [
                f"#{int(record['index']):03d} STEP={_step4(record.get('step_id', 0))} "
                f"MAP={_step4(record.get('grid_step', 0))} IMG={_step4(record.get('image_step', 0))}",
                "GOAL="
                f"({float(goal[0]):.2f},{float(goal[1]):.2f}) yaw={float(record.get('goal_yaw', 0.0)):.2f} "
                f"G={0 if global_plan is None else len(global_plan.get('poses', []))} "
                f"L={0 if local_plan is None else len(local_plan.get('poses', []))}",
            ]
            overlay_crop_path = self._render_overlay(
                Path(overlay_path),
                grid,
                pose,
                (float(goal[0]), float(goal[1])),
                record.get("trajectory_snapshot") or list(self.trajectory),
                global_plan=global_plan,
                local_plan=local_plan,
                goal_yaw=float(record.get("goal_yaw", 0.0)),
                label_lines=label_lines,
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
            step_id=int(record.get("step_id", 0)),
            grid_step=int(record.get("grid_step", 0)),
            image_step=int(record.get("image_step", 0)),
        )
        record["panel"] = panel_path

    def _plan_reaches_goal(self, plan: dict | None, goal_xy: tuple[float, float] | None) -> bool:
        if plan is None or goal_xy is None or not plan.get("poses"):
            return False
        last_pose = plan.get("poses", [])[-1]
        converted = self._transform_xy_yaw_to_frame(
            float(last_pose[0]),
            float(last_pose[1]),
            float(last_pose[2]) if len(last_pose) > 2 else 0.0,
            plan.get("frame_id", ""),
            self.args.map_frame,
        )
        if converted is None:
            return False
        tolerance = float(self.args.plan_goal_match_tolerance_m)
        return math.hypot(converted[0] - float(goal_xy[0]), converted[1] - float(goal_xy[1])) <= tolerance

    def _plan_matches_subgoal_window(
        self,
        plan: dict | None,
        goal_elapsed: float,
        max_pre: float,
        max_post: float,
        goal_xy: tuple[float, float] | None = None,
        require_goal_endpoint: bool = False,
    ) -> bool:
        if plan is None or not plan.get("poses"):
            return False
        plan_elapsed = float(plan.get("elapsed_sec", 0.0))
        if not (goal_elapsed - max_pre <= plan_elapsed <= goal_elapsed + max_post):
            return False
        if require_goal_endpoint and not self._plan_reaches_goal(plan, goal_xy):
            return False
        return True

    def _select_plan_for_subgoal_locked(
        self,
        plan_type: str,
        goal_elapsed: float,
        next_goal_elapsed: float | None,
        goal_xy: tuple[float, float] | None = None,
    ) -> dict | None:
        lower = goal_elapsed + float(self.args.plan_match_min_after_goal_sec)
        if next_goal_elapsed is None:
            upper = goal_elapsed + float(self.args.plan_match_post_goal_sec)
        else:
            upper = min(next_goal_elapsed - 1e-3, goal_elapsed + float(self.args.plan_match_post_goal_sec))
        candidates = [
            plan
            for plan in self.plan_records.get(plan_type, [])
            if plan.get("poses") and lower <= float(plan.get("elapsed_sec", 0.0)) <= upper
        ]
        if plan_type == "global":
            candidates = [plan for plan in candidates if self._plan_reaches_goal(plan, goal_xy)]
        if not candidates:
            return None
        # Use the newest path in the goal's active window so the overlay reflects
        # the final path move_base was trying to execute for that subgoal.
        return max(candidates, key=lambda plan: float(plan.get("elapsed_sec", 0.0)))

    def _refresh_all_subgoal_overlays_locked(self) -> int:
        refreshed = 0
        for idx, record in enumerate(self.subgoal_records):
            if not isinstance(record, dict):
                continue
            grid = record.get("grid_snapshot") or self.latest_grid
            if grid is None:
                continue
            goal = record.get("goal")
            if not goal or len(goal) < 2:
                continue
            goal_elapsed = float(record.get("elapsed_sec", 0.0))
            next_goal_elapsed = None
            if idx + 1 < len(self.subgoal_records):
                next_goal_elapsed = float(self.subgoal_records[idx + 1].get("elapsed_sec", 0.0))
            goal_xy = (float(goal[0]), float(goal[1]))
            global_plan = self._select_plan_for_subgoal_locked("global", goal_elapsed, next_goal_elapsed, goal_xy=goal_xy)
            local_plan = self._select_plan_for_subgoal_locked("local", goal_elapsed, next_goal_elapsed)
            if global_plan is None and local_plan is None:
                continue
            robot_pose = record.get("robot_pose")
            pose = None
            if isinstance(robot_pose, list) and len(robot_pose) >= 3:
                pose = (float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2]))
            overlay_path = record.get("overlay")
            if overlay_path:
                label_lines = [
                    f"#{int(record['index']):03d} STEP={_step4(record.get('step_id', 0))} "
                    f"MAP={_step4(record.get('grid_step', 0))} IMG={_step4(record.get('image_step', 0))}",
                    "GOAL="
                    f"({float(goal[0]):.2f},{float(goal[1]):.2f}) yaw={float(record.get('goal_yaw', 0.0)):.2f} "
                    f"G={0 if global_plan is None else len(global_plan.get('poses', []))} "
                    f"L={0 if local_plan is None else len(local_plan.get('poses', []))}",
                ]
                overlay_crop_path = self._render_overlay(
                    Path(overlay_path),
                    grid,
                    pose,
                    (float(goal[0]), float(goal[1])),
                    record.get("trajectory_snapshot") or list(self.trajectory),
                    global_plan=global_plan,
                    local_plan=local_plan,
                    goal_yaw=float(record.get("goal_yaw", 0.0)),
                    label_lines=label_lines,
                )
                record["overlay_crop"] = overlay_crop_path
            record["global_plan_points"] = 0 if global_plan is None else len(global_plan.get("poses", []))
            record["global_plan_stamp"] = 0.0 if global_plan is None else float(global_plan.get("stamp", 0.0))
            record["global_plan_message_index"] = 0 if global_plan is None else int(global_plan.get("message_index", 0))
            record["local_plan_points"] = 0 if local_plan is None else len(local_plan.get("poses", []))
            record["local_plan_stamp"] = 0.0 if local_plan is None else float(local_plan.get("stamp", 0.0))
            record["local_plan_message_index"] = 0 if local_plan is None else int(local_plan.get("message_index", 0))
            panel_path = self._render_subgoal_panel(
                int(record["index"]),
                float(record["elapsed_sec"]),
                float(record.get("stamp", 0.0)),
                float(record.get("first_person_stamp", 0.0)),
                record.get("overlay_crop") or record.get("overlay") or "",
                record.get("first_person") or "",
                step_id=int(record.get("step_id", 0)),
                grid_step=int(record.get("grid_step", 0)),
                image_step=int(record.get("image_step", 0)),
            )
            record["panel"] = panel_path
            refreshed += 1
        return refreshed

    def _render_uniform_subgoal_crops_locked(self) -> dict:
        loaded_records: list[tuple[dict, int, int, bytearray]] = []
        base_width = 0
        base_height = 0
        union: tuple[int, int, int, int] | None = None
        for record in self.subgoal_records:
            if not isinstance(record, dict):
                continue
            overlay_path = record.get("overlay") or ""
            if not overlay_path:
                continue
            loaded = _read_png(Path(overlay_path))
            if loaded is None:
                continue
            width, height, rgb = loaded
            if base_width <= 0:
                base_width = width
                base_height = height
            if width != base_width or height != base_height:
                continue
            bbox = _content_bbox(rgb, width, height, ignore_top_px=int(self.args.uniform_crop_ignore_top_px))
            if bbox is None:
                continue
            union = _union_bbox(union, bbox)
            loaded_records.append((record, width, height, rgb))

        if union is None or not loaded_records:
            return {"bbox": None, "count": 0}

        bbox = _expand_bbox(union, base_width, base_height, int(self.args.crop_margin_px))
        rendered = 0
        for record, width, height, rgb in loaded_records:
            crop_width, crop_height, crop_rgb = _crop_rgb_to_bbox(rgb, width, height, bbox)
            index = int(record.get("index", rendered + 1))
            overlay_path = self.uniform_overlay_dir / f"subgoal_{index:04d}.png"
            _write_png(overlay_path, crop_width, crop_height, crop_rgb)
            record["overlay_uniform_crop"] = str(overlay_path)
            goal = record.get("goal") if isinstance(record.get("goal"), list) else [0.0, 0.0]
            if len(goal) < 2:
                goal = [0.0, 0.0]
            title = (
                f"#{index:03d} STEP={_step4(record.get('step_id', 0))} "
                f"MAP={_step4(record.get('grid_step', 0))} IMG={_step4(record.get('image_step', 0))} "
                f"GOAL=({float(goal[0]):.2f},{float(goal[1]):.2f}) "
                f"G={int(record.get('global_plan_points') or 0)} L={int(record.get('local_plan_points') or 0)}"
            )
            titled_width, titled_height, titled_rgb = _make_titled_panel(
                (crop_width, crop_height, crop_rgb),
                title,
                title_height_px=self.args.panel_title_height_px,
            )
            titled_path = self.uniform_overlay_titled_dir / f"subgoal_{index:04d}_overlay_titled.png"
            _write_png(titled_path, titled_width, titled_height, titled_rgb)
            record["overlay_uniform_titled"] = str(titled_path)
            panel_path = self._render_subgoal_panel(
                index,
                float(record.get("elapsed_sec", 0.0)),
                float(record.get("stamp", 0.0)),
                float(record.get("first_person_stamp", 0.0)),
                str(overlay_path),
                record.get("first_person") or "",
                step_id=int(record.get("step_id", 0)),
                grid_step=int(record.get("grid_step", 0)),
                image_step=int(record.get("image_step", 0)),
                output_dir=self.uniform_panel_dir,
            )
            record["panel_uniform_crop"] = panel_path
            rendered += 1
        return {"bbox": list(bbox), "count": rendered}

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
            if self.shutting_down:
                return
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
            if self.shutting_down:
                return
            active_changed = active_goal != self.last_explore_active_goal
            if now - self.last_explore_status_time < self.args.explore_status_period_sec and not active_changed:
                return
            self.last_explore_status_time = now
            self.last_explore_active_goal = active_goal
            self._write_event(
                "explore_status",
                {"elapsed_sec": now - self.start_wall_time, "step_id": self.debug_step, "payload": payload},
            )

    def stall_timer_callback(self, _event) -> None:  # noqa: ANN001
        if self.shutting_down:
            return
        now = time.time()
        with self.lock:
            if self.shutting_down or self.latest_pose is None or self.stall_reference_xy is None:
                return
            if self.args.stall_snapshot_require_active_goal and self.last_explore_active_goal is None:
                return
            if now - self.stall_reference_time < self.args.stall_snapshot_sec:
                return
            if now - self.last_stall_snapshot_time < self.args.stall_snapshot_cooldown_sec:
                return
            moved = math.hypot(self.latest_pose[0] - self.stall_reference_xy[0], self.latest_pose[1] - self.stall_reference_xy[1])
            if moved > self.args.stall_snapshot_distance_m:
                return
            self._write_stall_snapshot_locked(now, moved)
            self.last_stall_snapshot_time = now

    def _write_stall_snapshot_locked(self, now: float, moved_m: float) -> None:
        self.stall_snapshot_count += 1
        snapshot_dir = self.stall_snapshot_dir / f"stall_{self.stall_snapshot_count:04d}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        pose = self.latest_pose
        goal_xy = self._active_goal_xy_locked()
        if goal_xy is None:
            goal_xy = pose[:2] if pose is not None else (0.0, 0.0)
        trajectory = list(self.trajectory)
        global_plan = self.latest_global_plan
        local_plan = self.latest_local_plan
        grids = [
            ("occ_map", self.latest_grid),
            ("global_costmap", self.latest_global_costmap),
            ("local_costmap", self.latest_local_costmap),
        ]
        maps: dict[str, dict] = {}
        overlays: dict[str, str] = {}
        path_checks: dict[str, dict] = {}
        for name, grid in grids:
            if grid is None:
                maps[name] = {"available": False}
                continue
            prefix = snapshot_dir / name
            _write_grid_pgm_yaml(prefix, grid)
            maps[name] = {
                "available": True,
                "topic": self._grid_topic_name(name),
                "frame_id": grid.header.frame_id or "",
                "stamp": grid.header.stamp.to_sec() if grid.header.stamp else 0.0,
                "width": int(grid.info.width),
                "height": int(grid.info.height),
                "resolution": float(grid.info.resolution),
                "origin": [
                    float(grid.info.origin.position.x),
                    float(grid.info.origin.position.y),
                    _grid_origin_yaw(grid),
                ],
                "pgm": str(prefix.with_suffix(".pgm")),
                "yaml": str(prefix.with_suffix(".yaml")),
            }
            overlay_path = snapshot_dir / f"{name}_path_overlay.png"
            crop_path = self._render_overlay(
                overlay_path,
                grid,
                pose,
                goal_xy,
                trajectory,
                global_plan=global_plan,
                local_plan=local_plan,
            )
            overlays[name] = crop_path or str(overlay_path)
            path_checks[name] = {
                "global": self._analyze_plan_against_grid(grid, global_plan),
                "local": self._analyze_plan_against_grid(grid, local_plan),
            }
        record = {
            "index": self.stall_snapshot_count,
            "step_id": self.debug_step,
            "grid_step": self.latest_grid_step,
            "image_step": self.latest_image_step,
            "elapsed_sec": now - self.start_wall_time,
            "wall_time": now,
            "stall_duration_sec": now - self.stall_reference_time,
            "stall_moved_m": moved_m,
            "stall_reference_xy": list(self.stall_reference_xy),
            "robot_pose": list(pose) if pose is not None else None,
            "active_goal": self.last_explore_active_goal,
            "goal_xy": list(goal_xy),
            "global_plan_points": 0 if global_plan is None else len(global_plan.get("poses", [])),
            "local_plan_points": 0 if local_plan is None else len(local_plan.get("poses", [])),
            "maps": maps,
            "overlays": overlays,
            "path_checks": path_checks,
        }
        (snapshot_dir / "snapshot_summary.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
        self.stall_snapshot_records.append(record)
        self._write_event("stall_snapshot", record)

    def _active_goal_xy_locked(self) -> tuple[float, float] | None:
        active = self.last_explore_active_goal
        if not isinstance(active, dict):
            return None
        point = active.get("point")
        if isinstance(point, list) and len(point) >= 2:
            return float(point[0]), float(point[1])
        return None

    def _active_goal_yaw_locked(self) -> float | None:
        active = self.last_explore_active_goal
        if not isinstance(active, dict):
            return None
        yaw = active.get("yaw")
        if yaw is None:
            return None
        return float(yaw)

    def _grid_topic_name(self, name: str) -> str:
        if name == "occ_map":
            return self.args.occupancy_grid_topic
        if name == "global_costmap":
            return self.args.global_costmap_topic
        if name == "local_costmap":
            return self.args.local_costmap_topic
        return ""

    def _analyze_plan_against_grid(self, grid: OccupancyGrid, plan: dict | None) -> dict:
        if plan is None or not plan.get("poses"):
            return {"available": False, "pose_count": 0, "sample_count": 0}
        poses = self._plan_poses_for_grid(grid, plan)
        if poses is None:
            return {
                "available": False,
                "reason": "transform_failed",
                "grid_frame_id": grid.header.frame_id or "",
                "plan_frame_id": plan.get("frame_id", ""),
                "pose_count": len(plan.get("poses", [])),
                "sample_count": 0,
            }
        resolution = max(float(grid.info.resolution), 1e-6)
        counts = {"free": 0, "occupied": 0, "unknown": 0, "out": 0}
        first_hits = []
        for start, end in zip(poses, poses[1:]):
            sx, sy = float(start[0]), float(start[1])
            ex, ey = float(end[0]), float(end[1])
            dist = math.hypot(ex - sx, ey - sy)
            samples = max(1, int(math.ceil(dist / (resolution * 0.5))))
            for index in range(samples + 1):
                ratio = float(index) / float(samples)
                x = sx * (1.0 - ratio) + ex * ratio
                y = sy * (1.0 - ratio) + ey * ratio
                cell = _world_to_cell(grid, x, y)
                value = _grid_value(grid, cell)
                if value is None:
                    label = "out"
                elif _is_unknown(value):
                    label = "unknown"
                elif _is_occupied(value):
                    label = "occupied"
                else:
                    label = "free"
                counts[label] += 1
                if label in ("occupied", "unknown") and len(first_hits) < self.args.path_check_max_hits:
                    first_hits.append(
                        {
                            "x": x,
                            "y": y,
                            "class": label,
                            "cell": None if cell is None else [cell[0], cell[1]],
                            "value": value,
                        }
                    )
        return {
            "available": True,
            "grid_frame_id": grid.header.frame_id or "",
            "plan_frame_id": plan.get("frame_id", ""),
            "frame_id": plan.get("frame_id", ""),
            "stamp": float(plan.get("stamp", 0.0)),
            "message_index": int(plan.get("message_index", 0)),
            "pose_count": len(poses),
            "sample_count": sum(counts.values()),
            "counts": counts,
            "first_hits": first_hits,
        }

    @staticmethod
    def _plan_frame_matches_grid(grid: OccupancyGrid, plan: dict | None) -> bool:
        if plan is None:
            return False
        plan_frame = str(plan.get("frame_id", "") or "")
        grid_frame = str(grid.header.frame_id or "")
        return not plan_frame or not grid_frame or plan_frame == grid_frame

    @staticmethod
    def _clean_frame_id(frame_id: str | None) -> str:
        return str(frame_id or "").lstrip("/")

    def _transform_xy_yaw_to_frame(
        self,
        x: float,
        y: float,
        yaw: float,
        source_frame: str | None,
        target_frame: str | None,
    ) -> tuple[float, float, float] | None:
        source = self._clean_frame_id(source_frame)
        target = self._clean_frame_id(target_frame)
        if not source or not target or source == target:
            return float(x), float(y), float(yaw)
        try:
            translation, rotation = self.tf_listener.lookupTransform(target, source, rospy.Time(0))
        except Exception:
            return None
        qx, qy, qz, qw = rotation
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        tf_yaw = math.atan2(siny_cosp, cosy_cosp)
        cos_yaw = math.cos(tf_yaw)
        sin_yaw = math.sin(tf_yaw)
        tx, ty = float(translation[0]), float(translation[1])
        return (
            tx + cos_yaw * float(x) - sin_yaw * float(y),
            ty + sin_yaw * float(x) + cos_yaw * float(y),
            float(yaw) + tf_yaw,
        )

    def _pose_to_grid_frame(
        self,
        grid: OccupancyGrid,
        pose: tuple[float, float, float] | None,
        source_frame: str | None,
    ) -> tuple[float, float, float] | None:
        if pose is None:
            return None
        return self._transform_xy_yaw_to_frame(pose[0], pose[1], pose[2], source_frame, grid.header.frame_id)

    def _point_to_grid_frame(
        self,
        grid: OccupancyGrid,
        point_xy: tuple[float, float] | None,
        source_frame: str | None,
        yaw: float = 0.0,
    ) -> tuple[float, float, float] | None:
        if point_xy is None:
            return None
        return self._transform_xy_yaw_to_frame(point_xy[0], point_xy[1], yaw, source_frame, grid.header.frame_id)

    def _plan_poses_for_grid(self, grid: OccupancyGrid, plan: dict | None) -> list[tuple[float, float, float]] | None:
        if plan is None or not plan.get("poses"):
            return None
        source_frame = plan.get("frame_id", "")
        transformed: list[tuple[float, float, float]] = []
        for pose in plan.get("poses", []):
            converted = self._transform_xy_yaw_to_frame(
                float(pose[0]),
                float(pose[1]),
                float(pose[2]) if len(pose) > 2 else 0.0,
                source_frame,
                grid.header.frame_id,
            )
            if converted is None:
                return None
            transformed.append(converted)
        return transformed

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
        goal_yaw: float | None = None,
        label_lines: list[str] | None = None,
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
        for y in range(height):
            for x in range(width):
                if _is_frontier_cell_data(grid.data, width, height, x, y):
                    py = height - 1 - y
                    index = (py * width + x) * 3
                    rgb[index : index + 3] = bytes((112, 36, 170))

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

        def draw_filled_triangle(points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
            if len(points) != 3:
                return
            min_x = max(0, min(point[0] for point in points))
            max_x = min(width - 1, max(point[0] for point in points))
            min_y = max(0, min(point[1] for point in points))
            max_y = min(height - 1, max(point[1] for point in points))
            x1, y1 = points[0]
            x2, y2 = points[1]
            x3, y3 = points[2]
            denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            if abs(denom) < 1e-6:
                return
            for py in range(min_y, max_y + 1):
                for px in range(min_x, max_x + 1):
                    a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
                    b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
                    c = 1.0 - a - b
                    if a >= 0.0 and b >= 0.0 and c >= 0.0:
                        draw_pixel(px, py, color)

        def draw_robot_arrow(px: int, py: int, yaw: float, length: int, color: tuple[int, int, int]) -> None:
            heading_x = math.cos(yaw)
            heading_y = -math.sin(yaw)
            perp_x = -heading_y
            perp_y = heading_x
            tip = (int(round(px + heading_x * length)), int(round(py + heading_y * length)))
            back_x = px - heading_x * length * 0.55
            back_y = py - heading_y * length * 0.55
            left = (int(round(back_x + perp_x * length * 0.45)), int(round(back_y + perp_y * length * 0.45)))
            right = (int(round(back_x - perp_x * length * 0.45)), int(round(back_y - perp_y * length * 0.45)))
            draw_filled_triangle([tip, left, right], color)
            outline = (0, 35, 160) if color == (0, 88, 255) else (150, 0, 20)
            for start, end in ((tip, left), (left, right), (right, tip)):
                for line_x, line_y in _bresenham(start[0], start[1], end[0], end[1]):
                    draw_pixel(line_x, line_y, outline)

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

        grid_frame = grid.header.frame_id or ""
        pose_in_grid = self._pose_to_grid_frame(grid, pose, self.args.odom_frame)
        goal_yaw_for_transform = 0.0 if goal_yaw is None else goal_yaw
        goal_in_grid = self._point_to_grid_frame(grid, goal_xy, self.args.map_frame, goal_yaw_for_transform)
        global_plan_poses = self._plan_poses_for_grid(grid, global_plan)
        local_plan_poses = self._plan_poses_for_grid(grid, local_plan)

        trajectory_pixels = []
        for _, x, y, _ in trajectory:
            transformed = self._transform_xy_yaw_to_frame(x, y, 0.0, self.args.odom_frame, grid_frame)
            if transformed is None:
                continue
            pixel = cell_to_pixel(_world_to_cell(grid, transformed[0], transformed[1]))
            if pixel is not None:
                trajectory_pixels.append(pixel)
        for start, end in zip(trajectory_pixels, trajectory_pixels[1:]):
            for px, py in _bresenham(start[0], start[1], end[0], end[1]):
                draw_pixel(px, py, (20, 118, 230))

        goal_cell = None if goal_in_grid is None else _world_to_cell(grid, goal_in_grid[0], goal_in_grid[1])
        if goal_cell is not None:
            radius_cells = max(1, int(math.ceil(self.args.frontier_check_radius_m / max(float(grid.info.resolution), 1e-6))))
            gx, gy = goal_cell
            for cy in range(gy - radius_cells, gy + radius_cells + 1):
                for cx in range(gx - radius_cells, gx + radius_cells + 1):
                    if cx < 0 or cy < 0 or cx >= width or cy >= height:
                        continue
                    if not _is_free(int(grid.data[cy * width + cx])):
                        continue
                    has_unknown_neighbor = False
                    for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                        if nx < 0 or ny < 0 or nx >= width or ny >= height:
                            continue
                        if _is_unknown(int(grid.data[ny * width + nx])):
                            has_unknown_neighbor = True
                            break
                    if has_unknown_neighbor:
                        pixel = cell_to_pixel((cx, cy))
                        if pixel is not None:
                            draw_circle(pixel[0], pixel[1], 2, (112, 36, 170))

        if global_plan_poses:
            draw_polyline(global_plan_poses, (40, 190, 60))
        if local_plan_poses:
            draw_polyline(local_plan_poses, (240, 150, 20))

        if pose_in_grid is not None:
            robot_pixel = cell_to_pixel(_world_to_cell(grid, pose_in_grid[0], pose_in_grid[1]))
            if robot_pixel is not None:
                draw_robot_arrow(robot_pixel[0], robot_pixel[1], pose_in_grid[2], 12, (0, 88, 255))

        goal_pixel = None if goal_in_grid is None else cell_to_pixel(_world_to_cell(grid, goal_in_grid[0], goal_in_grid[1]))
        if goal_pixel is not None and goal_in_grid is not None:
            draw_robot_arrow(goal_pixel[0], goal_pixel[1], goal_in_grid[2], 12, (230, 30, 45))
            draw_circle(goal_pixel[0], goal_pixel[1], 4, (230, 30, 45))

        if label_lines:
            box_height = min(height, 8 + len(label_lines) * 18)
            for py in range(0, box_height):
                for px in range(0, min(width, 760)):
                    draw_pixel(px, py, (255, 255, 255))
            for line_index, line in enumerate(label_lines):
                _draw_text(rgb, width, height, 8, 6 + line_index * 18, line, (15, 15, 15), scale=2)

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
        step_id: int = 0,
        grid_step: int = 0,
        image_step: int = 0,
        output_dir: Path | None = None,
    ) -> str:
        if not overlay_path:
            return ""
        left = _read_png(Path(overlay_path))
        if left is None:
            return ""
        right = _read_png(Path(first_person_path)) if first_person_path else None
        title = (
            f"#{index:03d} STEP={_step4(step_id)} MAP={_step4(grid_step)} IMG={_step4(image_step)} "
            f"T={elapsed_sec:.1f}S GOAL={goal_stamp:.3f} IMG_STAMP={image_stamp:.3f}"
        )
        panel_width, panel_height, panel_rgb = _make_side_by_side_panel(
            left,
            right,
            title=title,
            image_height=self.args.panel_image_height_px,
            gap_px=self.args.panel_gap_px,
            title_height_px=self.args.panel_title_height_px,
        )
        panel_path = (output_dir or self.panel_dir) / f"subgoal_{index:04d}_panel.png"
        _write_png(panel_path, panel_width, panel_height, panel_rgb)
        return str(panel_path)

    def _render_subgoal_contact_sheet(self) -> str:
        panels = []
        for record in self.subgoal_records:
            panel_path = ""
            if isinstance(record, dict):
                panel_path = record.get("panel_uniform_crop") or record.get("panel") or ""
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

    def _render_subgoal_overlay_contact_sheet(self) -> str:
        overlays = []
        for record in self.subgoal_records:
            overlay_path = ""
            if isinstance(record, dict):
                overlay_path = (
                    record.get("overlay_uniform_titled")
                    or record.get("overlay_uniform_crop")
                    or record.get("overlay_crop")
                    or record.get("overlay")
                    or ""
                )
            if not overlay_path:
                continue
            loaded = _read_png(Path(overlay_path))
            if loaded is not None:
                overlays.append(loaded)
        contact_sheet = _make_contact_sheet(
            overlays,
            columns=self.args.overlay_contact_sheet_columns,
            gap_px=self.args.contact_sheet_gap_px,
        )
        if contact_sheet is None:
            return ""
        width, height, rgb = contact_sheet
        path = self.output_dir / "subgoal_overlays_contact_sheet.png"
        _write_png(path, width, height, rgb)
        return str(path)

    @staticmethod
    def _json_safe_record(record: dict) -> dict:
        return {
            key: value
            for key, value in record.items()
            if key not in {"grid_snapshot", "trajectory_snapshot"}
        }

    def _json_safe_subgoals(self) -> list[dict]:
        return [self._json_safe_record(record) for record in self.subgoal_records if isinstance(record, dict)]

    def _write_event(self, event_type: str, payload: dict) -> None:
        row = {
            "type": event_type,
            "wall_time": time.time(),
            "step_id": self.debug_step,
            "elapsed_sec": time.time() - self.start_wall_time,
            **payload,
        }
        self.events_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def _render_semantic_contact_sheet(self) -> str:
        paths = sorted(self.semantic_keyframe_dir.glob("revision_*.png"))
        if len(paths) > 12:
            sample_indices = np.linspace(0, len(paths) - 1, 12, dtype=int) if np is not None else range(12)
            paths = [paths[int(index)] for index in sample_indices]
        panels = []
        for path in paths:
            loaded = _read_png(path)
            if loaded is None:
                continue
            width, height, rgb = loaded
            if cv2 is not None and np is not None:
                target_width = 900
                target_height = max(1, int(round(height * target_width / max(1, width))))
                image = np.frombuffer(bytes(rgb), dtype=np.uint8).reshape((height, width, 3))
                resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
                panels.append((target_width, target_height, bytearray(resized.tobytes())))
            else:
                panels.append(_resize_rgb_nearest(rgb, width, height, target_width=900))
        sheet = _make_contact_sheet(panels, columns=2, gap_px=12)
        if sheet is None:
            return ""
        width, height, rgb = sheet
        path = self.output_dir / "semantic_contact_sheet.png"
        _write_png(path, width, height, rgb)
        return str(path)

    def _semantic_summary(self) -> dict:
        graph = self.latest_unified_graph
        nodes = list(graph.get("nodes") or [])
        edges = list(graph.get("edges") or [])
        node_counts = {}
        for node in nodes:
            node_type = str(node.get("type", "object"))
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        portal_nodes = [node for node in nodes if node.get("type") == "portal"]
        return {
            "episode_id": graph.get("episode_id", ""),
            "graph_revision": int(graph.get("graph_revision", 0) or 0),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_counts": node_counts,
            "currently_visible_count": sum(bool(node.get("is_currently_visible")) for node in nodes),
            "connected_portal_count": sum(
                (node.get("attributes") or {}).get("connectivity_status") == "connected" for node in portal_nodes
            ),
            "partial_portal_count": sum(
                (node.get("attributes") or {}).get("connectivity_status") == "partial" for node in portal_nodes
            ),
            "contains_edge_count": sum(edge.get("relation") == "contains" for edge in edges),
            "semantic_event_count": len(self.semantic_events),
        }

    def shutdown(self) -> None:
        with self.video_lock:
            self._shutdown_locked()

    def _shutdown_locked(self) -> None:
        with self.lock:
            if self.shutting_down:
                return
            self.shutting_down = True
            try:
                self.stall_timer.shutdown()
            except Exception:
                pass
            for subscriber in self.subscribers:
                try:
                    subscriber.unregister()
                except Exception:
                    pass
            final_overlay = ""
            final_overlay_crop = ""
            final_first_person = ""
            final_external = ""
            if self.latest_image is not None:
                image_stamp, image_width, image_height, image_rgb = self.latest_image
                final_first_person = str(self.first_person_dir / "final_first_person.png")
                _write_png(Path(final_first_person), image_width, image_height, image_rgb)
                self.final_first_person_path = final_first_person
            if self.latest_external_image is not None:
                _external_stamp, external_width, external_height, external_rgb = self.latest_external_image
                final_external = str(self.external_dir / "final_external_camera.png")
                _write_png(Path(final_external), external_width, external_height, external_rgb)
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
            refreshed_subgoal_overlays = self._refresh_all_subgoal_overlays_locked()
            uniform_subgoal_crop = self._render_uniform_subgoal_crops_locked()
            subgoal_contact_sheet = self._render_subgoal_contact_sheet()
            subgoal_overlay_contact_sheet = self._render_subgoal_overlay_contact_sheet()
            semantic_contact_sheet = self._render_semantic_contact_sheet()
            graph_final = ""
            if self.latest_unified_graph:
                episode_id = str(self.latest_unified_graph.get("episode_id") or "episode")
                graph_final_path = self.graph_dir / f"{episode_id}_final.json"
                graph_final_path.write_text(
                    json.dumps(self.latest_unified_graph, ensure_ascii=False, indent=2, sort_keys=True)
                )
                graph_final = str(graph_final_path)
            with self.video_lock:
                self._finalize_first_person_video_locked()
                self._finalize_external_video_locked()
            summary = {
                "duration_sec": time.time() - self.start_wall_time,
                "final_step_id": self.debug_step,
                "final_grid_step": self.latest_grid_step,
                "final_image_step": self.latest_image_step,
                "distance_m": self.distance_m,
                "trajectory_samples": len(self.trajectory),
                "subgoal_count": self.goal_count,
                "current_subgoal_count": self.current_subgoal_count,
                "status_counts": self.status_counts,
                "cmd_vel_counts": self.cmd_vel_counts,
                "cmd_vel_nonzero_counts": self.cmd_vel_nonzero_counts,
                "cmd_vel_max_speed": self.cmd_vel_max_speed,
                "plan_message_counts": self.plan_message_counts,
                "stall_snapshot_count": self.stall_snapshot_count,
                "stall_snapshots": self.stall_snapshot_records,
                "first_pose": list(self.trajectory[0][1:]) if self.trajectory else None,
                "last_pose": list(self.trajectory[-1][1:]) if self.trajectory else None,
                "subgoals": self._json_safe_subgoals(),
                "final_overlay": final_overlay,
                "final_overlay_crop": final_overlay_crop,
                "final_first_person": final_first_person,
                "final_external_camera": final_external,
                "final_first_person_stamp": 0.0 if self.latest_image is None else float(self.latest_image[0]),
                "first_person_video": self.first_person_video_path
                if self.first_person_video_frame_count > 0
                else "",
                "first_person_video_raw": self.first_person_video_raw_path
                if Path(self.first_person_video_raw_path).exists()
                else "",
                "first_person_video_frame_count": self.first_person_video_frame_count,
                "first_person_video_fps": self.args.first_person_video_fps,
                "first_person_video_capture_fps": self.args.first_person_video_capture_fps,
                "first_person_video_codec": self.first_person_video_codec_name,
                "first_person_video_error": self.first_person_video_error,
                "first_person_video_map_mode": "latest_occ_snapshot_on_image_callback",
                "semantic_video": bool(self.args.semantic_video),
                "semantic_summary": self._semantic_summary(),
                "semantic_graph_final": graph_final,
                "semantic_events_jsonl": str(self.graph_dir / "graph_revision_events.jsonl"),
                "semantic_keyframes": str(self.semantic_keyframe_dir),
                "semantic_contact_sheet": semantic_contact_sheet,
                "first_person_video_map_max_age_sec": self.args.video_map_max_age_sec,
                "first_person_video_frames_csv": str(self.output_dir / "video_frames.csv"),
                "first_person_video_camera_frames": str(self.video_camera_frame_dir),
                "first_person_video_map_frames": str(self.video_map_frame_dir),
                "first_person_video_global_costmap_frames": str(self.video_global_costmap_frame_dir),
                "first_person_video_local_costmap_frames": str(self.video_local_costmap_frame_dir),
                "first_person_video_composite_frames": str(self.video_composite_frame_dir),
                "external_video": self.external_video_path
                if self.external_video_frame_count > 0
                else "",
                "external_video_raw": self.external_video_raw_path
                if Path(self.external_video_raw_path).exists()
                else "",
                "external_video_frame_count": self.external_video_frame_count,
                "external_video_codec": self.external_video_codec_name,
                "external_video_error": self.external_video_error,
                "external_video_frames": str(self.video_external_frame_dir),
                "refreshed_subgoal_overlays": refreshed_subgoal_overlays,
                "uniform_subgoal_crop": uniform_subgoal_crop,
                "subgoal_contact_sheet": subgoal_contact_sheet,
                "subgoal_overlay_contact_sheet": subgoal_overlay_contact_sheet,
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
                self.video_frames_file,
                self.move_base_log_file,
                self.semantic_events_file,
            ]:
                handle.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record explore_py runtime debug artifacts.")
    parser.add_argument("--output-dir", default="", help="Directory for JSONL, CSV, and PNG overlays.")
    parser.add_argument("--occupancy-grid-topic", default="/struct_mapping/occ_map")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--map-frame", default="tf_frame_map")
    parser.add_argument("--odom-frame", default="tf_frame_odom")
    parser.add_argument("--goal-topic", default="/move_base_simple/goal")
    parser.add_argument("--current-subgoal-topic", default="/explore_py/current_subgoal")
    parser.add_argument("--move-base-status-topic", default="/move_base/status")
    parser.add_argument("--explore-status-topic", default="/explore_py/status")
    parser.add_argument("--gt-observations-topic", default="/semantic_mapping/gt_observations")
    parser.add_argument("--unified-graph-topic", default="/semantic_mapping/unified_graph")
    parser.add_argument("--global-plan-topic", default="/move_base/GlobalPlanner/plan")
    parser.add_argument("--local-global-plan-topic", default="/move_base/DWAPlannerROS/global_plan")
    parser.add_argument("--local-plan-topic", default="/move_base/DWAPlannerROS/local_plan")
    parser.add_argument("--global-costmap-topic", default="/move_base/global_costmap/costmap")
    parser.add_argument("--local-costmap-topic", default="/move_base/local_costmap/costmap")
    parser.add_argument("--global-costmap-updates-topic", default="/move_base/global_costmap/costmap_updates")
    parser.add_argument("--local-costmap-updates-topic", default="/move_base/local_costmap/costmap_updates")
    parser.add_argument("--rosout-topic", default="/rosout_agg")
    parser.add_argument("--image-topic", default="/molmo_spaces/head_camera/image")
    parser.add_argument("--external-image-topic", default="")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--cmd-vel-stamped-topic", default="/cmd_vel_stamped")
    parser.add_argument("--frontier-check-radius-m", type=float, default=1.0)
    parser.add_argument("--trajectory-period-sec", type=float, default=0.5)
    parser.add_argument("--trajectory-min-step-m", type=float, default=0.02)
    parser.add_argument("--explore-status-period-sec", type=float, default=2.0)
    parser.add_argument("--max-odom-jump-m", type=float, default=3.0)
    parser.add_argument("--crop-margin-px", type=int, default=40)
    parser.add_argument("--crop-scale", type=int, default=4)
    parser.add_argument("--uniform-crop-ignore-top-px", type=int, default=80)
    parser.add_argument("--panel-image-height-px", type=int, default=520)
    parser.add_argument("--panel-title-height-px", type=int, default=34)
    parser.add_argument("--panel-gap-px", type=int, default=12)
    parser.add_argument("--contact-sheet-columns", type=int, default=1)
    parser.add_argument("--overlay-contact-sheet-columns", type=int, default=4)
    parser.add_argument("--contact-sheet-gap-px", type=int, default=16)
    parser.add_argument("--cmd-vel-record-period-sec", type=float, default=0.2)
    parser.add_argument("--cmd-vel-nonzero-threshold", type=float, default=1e-4)
    parser.add_argument("--stall-check-period-sec", type=float, default=1.0)
    parser.add_argument("--stall-snapshot-sec", type=float, default=30.0)
    parser.add_argument("--stall-snapshot-distance-m", type=float, default=0.15)
    parser.add_argument("--stall-snapshot-cooldown-sec", type=float, default=45.0)
    parser.add_argument("--stall-snapshot-require-active-goal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--path-check-max-hits", type=int, default=25)
    parser.add_argument("--first-person-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--first-person-video-with-map", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--semantic-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--external-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--first-person-video-fps", type=float, default=15.0)
    parser.add_argument(
        "--first-person-video-capture-fps",
        type=float,
        default=1.0,
        help="Online six-panel render rate; captured frames are each written once at --first-person-video-fps.",
    )
    parser.add_argument("--first-person-video-width-px", type=int, default=960)
    parser.add_argument("--first-person-video-codec", default="mp4v")
    parser.add_argument("--first-person-video-h264", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--first-person-video-h264-crf", type=int, default=23)
    parser.add_argument("--first-person-video-h264-preset", default="veryfast")
    parser.add_argument("--first-person-video-h264-timeout-sec", type=float, default=180.0)
    parser.add_argument("--video-map-crop-margin-px", type=int, default=90)
    parser.add_argument("--video-map-desync-step-warn", type=int, default=3)
    parser.add_argument("--video-map-max-age-sec", type=float, default=2.0)
    parser.add_argument("--video-sync-max-delta-sec", type=float, default=0.05)
    parser.add_argument("--video-stuck-window-sec", type=float, default=20.0)
    parser.add_argument("--video-stuck-distance-m", type=float, default=0.15)
    parser.add_argument("--video-stuck-rotation-yaw-rad", type=float, default=0.35)
    parser.add_argument("--plan-match-pre-goal-sec", type=float, default=2.0)
    parser.add_argument("--plan-match-min-after-goal-sec", type=float, default=0.0)
    parser.add_argument("--plan-match-post-goal-sec", type=float, default=60.0)
    parser.add_argument("--plan-goal-match-tolerance-m", type=float, default=1.0)
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
