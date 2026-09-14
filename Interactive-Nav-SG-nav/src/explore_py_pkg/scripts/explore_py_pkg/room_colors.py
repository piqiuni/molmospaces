"""Stable room-ID colors shared by live and offline recording."""

import colorsys


def room_color(room_id: int) -> tuple[int, int, int]:
    # Golden-angle hues do not repeat every eight IDs or change with visibility.
    hue = (int(room_id) * 0.6180339887498949) % 1.0
    return tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, 0.43, 0.96))
