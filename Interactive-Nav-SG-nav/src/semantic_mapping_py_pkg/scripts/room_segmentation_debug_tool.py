#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR / "semantic_mapping_py_pkg"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from semantic_mapping_py_pkg.room_segmentation import RoomSegmenter, RoomSegmentationState
from semantic_mapping_py_pkg.ros_py311_compat import patch_roslogging_findcaller_for_py311


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Save and replay occupancy-grid room segmentation snapshots.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    live = subparsers.add_parser("live", help="Subscribe to an occupancy-grid topic and save snapshots on change.")
    live.add_argument("--topic", default="/struct_mapping/occ_map")
    live.add_argument("--output-dir", required=True)
    live.add_argument("--min-changed-cells", type=int, default=500)
    live.add_argument("--min-changed-ratio", type=float, default=0.02)

    replay = subparsers.add_parser("replay", help="Replay saved occupancy grids and render room overlays.")
    replay.add_argument("--input", required=True, help="Snapshot file or directory containing .npz snapshots.")
    replay.add_argument("--output-dir", required=True)
    replay.add_argument("--room-free-threshold", type=int, default=20)
    replay.add_argument("--room-unknown-id", type=int, default=-1)
    replay.add_argument("--room-min-component-cells", type=int, default=25)
    replay.add_argument("--room-boundary-margin-cells", type=int, default=1)
    replay.add_argument("--room-core-min-component-cells", type=int, default=40)
    replay.add_argument("--room-core-clearance-cells", type=int, default=7)
    replay.add_argument("--room-small-obstacle-max-cells", type=int, default=0)
    replay.add_argument("--room-remove-enclosed-occupied", dest="room_remove_enclosed_occupied", action="store_true")
    replay.add_argument("--no-room-remove-enclosed-occupied", dest="room_remove_enclosed_occupied", action="store_false")
    replay.set_defaults(room_remove_enclosed_occupied=True)
    replay.add_argument("--room-enclosed-occupied-max-cells", type=int, default=800)
    replay.add_argument("--room-enclosed-occupied-max-aspect", type=float, default=3.0)
    replay.add_argument("--room-enclosed-occupied-known-ring-ratio", type=float, default=0.95)
    replay.add_argument("--room-enclosed-occupied-free-ring-ratio", type=float, default=0.4)
    replay.add_argument("--room-fill-enclosed-obstacles", action="store_true")
    replay.add_argument("--room-enclosed-obstacle-min-cells", type=int, default=120)
    replay.add_argument("--room-enclosed-obstacle-max-cells", type=int, default=700)
    replay.add_argument("--room-enclosed-obstacle-dominance-ratio", type=float, default=0.82)
    replay.add_argument("--crop-padding-cells", type=int, default=12)
    replay.add_argument("--min-output-size", type=int, default=960)
    return parser


def occupancy_counts(data):
    arr = np.asarray(data, dtype=np.int16)
    return {
        "free": int(np.sum((arr >= 0) & (arr <= 20))),
        "occupied": int(np.sum(arr > 20)),
        "unknown": int(np.sum(arr < 0)),
    }


def changed_cells(current, previous):
    current_arr = np.asarray(current, dtype=np.int16)
    previous_arr = np.asarray(previous, dtype=np.int16)
    if current_arr.shape != previous_arr.shape:
        return current_arr.size
    return int(np.sum(current_arr != previous_arr))


def save_snapshot(msg, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp_ns = int(time.time() * 1e9)
    path = output_dir / f"occ_snapshot_{stamp_ns}.npz"
    np.savez_compressed(
        path,
        data=np.asarray(msg.data, dtype=np.int16),
        width=int(msg.info.width),
        height=int(msg.info.height),
        resolution=float(msg.info.resolution),
        origin_x=float(msg.info.origin.position.x),
        origin_y=float(msg.info.origin.position.y),
        origin_z=float(msg.info.origin.position.z),
        frame_id=str(msg.header.frame_id),
        stamp_secs=int(msg.header.stamp.secs),
        stamp_nsecs=int(msg.header.stamp.nsecs),
    )
    return path


def run_live(args):
    patch_roslogging_findcaller_for_py311()
    import rospy
    from nav_msgs.msg import OccupancyGrid

    output_dir = Path(args.output_dir).resolve()
    state = {"last_data": None, "last_path": None}

    def callback(msg):
        current = list(msg.data)
        previous = state["last_data"]
        should_save = previous is None
        if not should_save:
            diff_cells = changed_cells(current, previous)
            total = max(len(current), 1)
            diff_ratio = float(diff_cells) / float(total)
            should_save = diff_cells >= args.min_changed_cells or diff_ratio >= args.min_changed_ratio
        if not should_save:
            return
        path = save_snapshot(msg, output_dir)
        counts = occupancy_counts(current)
        rospy.loginfo(
            "[room_segmentation_debug_tool] saved %s free=%d occupied=%d unknown=%d",
            path,
            counts["free"],
            counts["occupied"],
            counts["unknown"],
        )
        state["last_data"] = current
        state["last_path"] = str(path)

    rospy.init_node("room_segmentation_debug_tool")
    rospy.Subscriber(args.topic, OccupancyGrid, callback, queue_size=1)
    rospy.loginfo(
        "[room_segmentation_debug_tool] live mode topic=%s output_dir=%s min_changed_cells=%d min_changed_ratio=%.4f",
        args.topic,
        str(output_dir),
        int(args.min_changed_cells),
        float(args.min_changed_ratio),
    )
    rospy.spin()


class SnapshotOccGrid:
    class _Header:
        class _Stamp:
            def __init__(self, secs, nsecs):
                self.secs = int(secs)
                self.nsecs = int(nsecs)

        def __init__(self, frame_id, secs, nsecs):
            self.frame_id = str(frame_id)
            self.stamp = SnapshotOccGrid._Header._Stamp(secs, nsecs)

    class _OriginPosition:
        def __init__(self, x, y, z):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class _Origin:
        def __init__(self, x, y, z):
            self.position = SnapshotOccGrid._OriginPosition(x, y, z)

    class _Info:
        def __init__(self, width, height, resolution, origin_x, origin_y, origin_z):
            self.width = int(width)
            self.height = int(height)
            self.resolution = float(resolution)
            self.origin = SnapshotOccGrid._Origin(origin_x, origin_y, origin_z)

    def __init__(self, payload):
        self.data = payload["data"].astype(np.int16).tolist()
        self.info = SnapshotOccGrid._Info(
            payload["width"],
            payload["height"],
            payload["resolution"],
            payload["origin_x"],
            payload["origin_y"],
            payload["origin_z"],
        )
        self.header = SnapshotOccGrid._Header(
            payload["frame_id"],
            payload["stamp_secs"],
            payload["stamp_nsecs"],
        )


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as payload:
        return SnapshotOccGrid(payload)


def occupancy_to_rgb(data, width, height):
    arr = np.asarray(data, dtype=np.int16).reshape(height, width)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[arr < 0] = np.array([40, 40, 40], dtype=np.uint8)
    rgb[(arr >= 0) & (arr <= 20)] = np.array([245, 245, 245], dtype=np.uint8)
    rgb[arr > 20] = np.array([20, 20, 20], dtype=np.uint8)
    return rgb


def room_overlay_rgb(base_rgb, room_ids, width, height, unknown_id):
    overlay = np.array(base_rgb, copy=True)
    arr = np.asarray(room_ids, dtype=np.int32).reshape(height, width)
    palette = np.asarray(
        [
            [239, 83, 80],
            [66, 165, 245],
            [102, 187, 106],
            [255, 202, 40],
            [171, 71, 188],
            [255, 112, 67],
            [38, 198, 218],
            [141, 110, 99],
        ],
        dtype=np.uint8,
    )
    valid_mask = arr != int(unknown_id)
    unique_rooms = sorted(int(room_id) for room_id in np.unique(arr[valid_mask])) if np.any(valid_mask) else []
    for room_id in unique_rooms:
        room_mask = arr == room_id
        color = palette[room_id % len(palette)]
        overlay[room_mask] = (0.55 * overlay[room_mask] + 0.45 * color).astype(np.uint8)
    return overlay


def content_bbox(data, width, height, padding_cells):
    arr = np.asarray(data, dtype=np.int16).reshape(height, width)
    mask = arr >= 0
    if not np.any(mask):
        return 0, height, 0, width
    ys, xs = np.where(mask)
    row_min = max(int(np.min(ys)) - int(padding_cells), 0)
    row_max = min(int(np.max(ys)) + int(padding_cells) + 1, height)
    col_min = max(int(np.min(xs)) - int(padding_cells), 0)
    col_max = min(int(np.max(xs)) + int(padding_cells) + 1, width)
    return row_min, row_max, col_min, col_max


def crop_image(image, bbox):
    row_min, row_max, col_min, col_max = bbox
    return np.asarray(image[row_min:row_max, col_min:col_max], copy=True)


def upscale_image(image, min_output_size):
    height, width = image.shape[:2]
    longest = max(height, width, 1)
    scale = max(1, int(np.ceil(float(min_output_size) / float(longest))))
    if scale <= 1:
        return image
    try:
        from PIL import Image

        pil_image = Image.fromarray(image)
        resized = pil_image.resize((width * scale, height * scale), resample=Image.NEAREST)
        return np.asarray(resized)
    except Exception:
        import cv2

        return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)


def save_overlay_image(path, image):
    try:
        from PIL import Image

        Image.fromarray(image).save(path)
        return
    except Exception:
        import cv2

        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def iter_snapshot_paths(input_path):
    input_path = Path(input_path).resolve()
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.npz"))


def run_replay(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = RoomSegmentationState()
    segmenter = RoomSegmenter(
        room_free_threshold=args.room_free_threshold,
        room_unknown_id=args.room_unknown_id,
        room_min_component_cells=args.room_min_component_cells,
        room_boundary_margin_cells=args.room_boundary_margin_cells,
        room_core_min_component_cells=args.room_core_min_component_cells,
        room_core_clearance_cells=args.room_core_clearance_cells,
        room_small_obstacle_max_cells=args.room_small_obstacle_max_cells,
        room_remove_enclosed_occupied=args.room_remove_enclosed_occupied,
        room_enclosed_occupied_max_cells=args.room_enclosed_occupied_max_cells,
        room_enclosed_occupied_max_aspect=args.room_enclosed_occupied_max_aspect,
        room_enclosed_occupied_known_ring_ratio=args.room_enclosed_occupied_known_ring_ratio,
        room_enclosed_occupied_free_ring_ratio=args.room_enclosed_occupied_free_ring_ratio,
        room_fill_enclosed_obstacles=args.room_fill_enclosed_obstacles,
        room_enclosed_obstacle_min_cells=args.room_enclosed_obstacle_min_cells,
        room_enclosed_obstacle_max_cells=args.room_enclosed_obstacle_max_cells,
        room_enclosed_obstacle_dominance_ratio=args.room_enclosed_obstacle_dominance_ratio,
        state=state,
    )
    summary = []
    for snapshot_path in iter_snapshot_paths(args.input):
        occ_grid = load_snapshot(snapshot_path)
        room_ids, room_conf = segmenter.segment(occ_grid)
        width = int(occ_grid.info.width)
        height = int(occ_grid.info.height)
        base_rgb = occupancy_to_rgb(occ_grid.data, width, height)
        overlay = room_overlay_rgb(base_rgb, room_ids, width, height, args.room_unknown_id)
        bbox = content_bbox(occ_grid.data, width, height, args.crop_padding_cells)
        overlay = crop_image(overlay, bbox)
        overlay = upscale_image(overlay, args.min_output_size)
        output_image = output_dir / f"{snapshot_path.stem}_room_overlay.png"
        save_overlay_image(output_image, overlay)
        room_count = len({int(room_id) for room_id in room_ids if int(room_id) >= 0})
        unique_conf = sorted(int(v) for v in set(int(conf) for conf in room_conf if int(conf) >= 0))
        summary.append(
            {
                "snapshot": str(snapshot_path),
                "overlay_image": str(output_image),
                "room_count": int(room_count),
                "width": width,
                "height": height,
                "crop_bbox_rc": [int(v) for v in bbox],
                "output_height": int(overlay.shape[0]),
                "output_width": int(overlay.shape[1]),
                "confidence_levels": unique_conf,
            }
        )
        print(
            f"[room_segmentation_debug_tool] wrote {output_image} rooms={room_count} "
            f"crop=({bbox[0]}:{bbox[1]}, {bbox[2]}:{bbox[3]}) size={overlay.shape[1]}x{overlay.shape[0]}"
        )
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"[room_segmentation_debug_tool] summary saved to {summary_path}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.mode == "live":
        run_live(args)
        return
    if args.mode == "replay":
        run_replay(args)
        return
    parser.error(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
