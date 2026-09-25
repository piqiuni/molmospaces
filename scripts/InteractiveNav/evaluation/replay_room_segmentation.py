"""Replay saved occupancy and public portal observations without ROS or simulation."""

import argparse
import gzip
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg"
sys.path.insert(0, str(PACKAGE / "scripts"))
from semantic_mapping_py_pkg.room_segmentation import RoomSegmenter
from semantic_mapping_py_pkg.semantic_occ_overlay import SemanticOccupancyOverlay


def read_grid(record):
    origin = record["origin"]
    info = NS(width=record["width"], height=record["height"], resolution=record["resolution"],
              origin=NS(position=NS(**{a: origin[a] for a in "xyz"}),
                        orientation=NS(**{a: origin['q' + a] for a in "xyzw"})))
    values = cv2.imread(record["image"], cv2.IMREAD_UNCHANGED)
    if values is None:
        raise FileNotFoundError(record["image"])
    values = values.astype(np.int32) - record["png_value_offset"]
    return NS(info=info, data=values.ravel().tolist()), values


def run(domain):
    config = yaml.safe_load((PACKAGE / "config/default.yaml").read_text())["semantic_map"]
    parameters = inspect.signature(RoomSegmenter).parameters
    segmenter = RoomSegmenter(**{k: v for k, v in config.items() if k in parameters})
    overlay = SemanticOccupancyOverlay()
    maps = [json.loads(line) for line in (domain / "debug/raw/map_manifest.jsonl").open()]
    raw_maps = {r["step_index"]: r for r in maps if r["stage"] == "raw_occ"}
    historical = next(r for r in reversed(maps) if r["stage"] == "room_segmentation")
    timeline = []
    last = None
    manifest = domain / "debug/raw/step_boundaries.jsonl"
    rows = manifest.open() if manifest.exists() else gzip.open(str(manifest) + ".gz", "rt")
    for line in rows:
        frame = json.loads(line)
        step = frame["step_index"]
        observations = (frame.get("gt_observations") or {}).get("observations") or []
        segmenter.update_portal_hints(observations, source_mode="realtime_gt_observation")
        if step not in raw_maps:
            continue
        grid, raw = read_grid(raw_maps[step])
        graph = frame.get("unified_graph") or {}
        overlay.update_graph(graph)
        if overlay.has_active_portals(include_pending=False):
            grid.data, _, _ = overlay.apply(grid.info, grid.data, include_pending=False)
        labels, confidence = segmenter.segment(grid)
        labels = np.asarray(labels).reshape(raw.shape)
        timeline.append({"step": step, "room_ids": np.unique(labels[labels >= 0]).tolist(),
                         "portal_hints": len(segmenter.state.portal_hints)})
        last = (grid, raw, labels)
    if last is None:
        raise ValueError(f"No replayable frames in {domain}")
    return last, read_grid(historical), timeline


def partition_agreement(replay, replay_info, recorded, recorded_info):
    """Compare partitions on shared labeled cells without assuming equal room IDs."""
    from scipy.optimize import linear_sum_assignment
    from semantic_mapping_py_pkg.geometry_utils import grid_origin_yaw

    rows, cols = np.nonzero(recorded >= 0)
    source_yaw, destination_yaw = grid_origin_yaw(recorded_info), grid_origin_yaw(replay_info)
    x = (cols + .5) * recorded_info.resolution
    y = (rows + .5) * recorded_info.resolution
    world_x = recorded_info.origin.position.x + np.cos(source_yaw) * x - np.sin(source_yaw) * y
    world_y = recorded_info.origin.position.y + np.sin(source_yaw) * x + np.cos(source_yaw) * y
    dx, dy = world_x - replay_info.origin.position.x, world_y - replay_info.origin.position.y
    dest_cols = np.floor((np.cos(destination_yaw) * dx + np.sin(destination_yaw) * dy) / replay_info.resolution).astype(int)
    dest_rows = np.floor((-np.sin(destination_yaw) * dx + np.cos(destination_yaw) * dy) / replay_info.resolution).astype(int)
    valid = (dest_cols >= 0) & (dest_cols < replay.shape[1]) & (dest_rows >= 0) & (dest_rows < replay.shape[0])
    left, right = replay[dest_rows[valid], dest_cols[valid]], recorded[rows[valid], cols[valid]]
    valid = left >= 0
    left, right = left[valid], right[valid]
    if not len(left):
        return {"shared_labeled_cells": 0, "label_permutation_invariant_agreement_on_shared_labeled_cells": None}
    _, a = np.unique(left, return_inverse=True)
    _, b = np.unique(right, return_inverse=True)
    counts = np.zeros((a.max() + 1, b.max() + 1), dtype=int)
    np.add.at(counts, (a, b), 1)
    row_ids, col_ids = linear_sum_assignment(-counts)
    return {"shared_labeled_cells": len(left),
            "label_permutation_invariant_agreement_on_shared_labeled_cells": float(counts[row_ids, col_ids].sum() / len(left))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    report = {"scope": "Fixed historical input diagnostic replay; source-step order, not exact ROS callback order. Uses repository semantic_map config and confirmed graph overlay. Not a closed-loop navigation evaluation.", "domains": {}}
    for row, name in enumerate(["channel", "container", "mixed"]):
        (grid, raw, labels), (old_grid, old_labels), timeline = run(args.run_dir / name)
        for col, (values, info, title) in enumerate([
            (raw, grid.info, "Recorded occupancy"),
            (old_labels, old_grid.info, "Recorded room labels"),
            (labels, grid.info, "Offline room replay")]):
            ax = axes[row, col]
            x, y, res = info.origin.position.x, info.origin.position.y, info.resolution
            shown = np.ma.masked_less(values, 0)
            ax.imshow(shown, origin="lower", interpolation="nearest", cmap="gray_r" if col == 0 else "tab20",
                      extent=[x, x + info.width * res, y, y + info.height * res])
            known = np.argwhere(values >= 0)
            if len(known):
                lo, hi = known.min(axis=0), known.max(axis=0) + 1
                ax.set_xlim(x + (lo[1] - 2) * res, x + (hi[1] + 2) * res)
                ax.set_ylim(y + (lo[0] - 2) * res, y + (hi[0] + 2) * res)
            ax.set_title(f"{name}: {title}"); ax.set_xlabel("world x (m)"); ax.set_ylabel("world y (m)")
        report["domains"][name] = {"replayed_frames": len(timeline), "final": timeline[-1],
            "recorded_room_ids": np.unique(old_labels[old_labels >= 0]).tolist()}
        report["domains"][name].update(partition_agreement(labels, grid.info, old_labels, old_grid.info))
        (args.output_dir / f"{name}_timeline.json").write_text(json.dumps(timeline, indent=2))
        np.savez_compressed(args.output_dir / f"{name}_final.npz", raw=raw, replay_labels=labels, recorded_labels=old_labels)
        print(name, report["domains"][name], flush=True)
    fig.savefig(args.output_dir / "room_replay_montage.png", dpi=150)
    report["comparison_note"] = "Agreement compares historical and replayed segmentation, not GT room accuracy; final recorded maps can be one frame newer."
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
