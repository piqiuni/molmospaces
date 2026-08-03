#!/usr/bin/env python3
"""Offline benchmark for the current room-segmentation hot path.

This intentionally does not import or mutate the ROS node.  It benchmarks the
current ``RoomSegmenter.segment`` implementation on a saved raw OCC map (or a
same-scale synthetic map when no raw map is available), then isolates two safe
optimization candidates:

* OpenCV connected-component labelling backends (WU/GRANA/BBDT/SAUF/
  SPAGHETTI) for the core-room mask.
* Replacing the current per-component ``np.where`` + Python list seed write
  with an equivalent vectorised label lookup table.

The candidates are microbenchmarks, not runtime substitutions.  They produce
enough evidence to choose the first low-risk runtime change without changing
room topology semantics as part of a timing experiment.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SEMANTIC_MAPPING_SCRIPTS = (
    REPO_ROOT
    / "Interactive-Nav-SG-nav"
    / "src"
    / "semantic_mapping_py_pkg"
    / "scripts"
)
if str(SEMANTIC_MAPPING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_MAPPING_SCRIPTS))

from semantic_mapping_py_pkg.room_segmentation import RoomSegmenter  # noqa: E402


DEFAULT_RAW_PGM = (
    REPO_ROOT
    / "outputs"
    / "occ_component_timing_smoke_100step_20260802"
    / "house_0006"
    / "debug"
    / "final_raw_occ_map.pgm"
)


def _origin() -> SimpleNamespace:
    return SimpleNamespace(
        position=SimpleNamespace(x=-100.0, y=-100.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _grid(values: np.ndarray, resolution: float = 0.1) -> SimpleNamespace:
    height, width = values.shape
    return SimpleNamespace(
        info=SimpleNamespace(
            width=int(width),
            height=int(height),
            resolution=float(resolution),
            origin=_origin(),
        ),
        data=values.reshape(-1).astype(np.int16).tolist(),
    )


def _load_ros_pgm(path: Path) -> np.ndarray:
    """Decode a ROS map_saver PGM into OccupancyGrid-compatible values.

    map_saver writes 254 for free, 0 for occupied, and 205 for unknown.  For
    known values its grayscale scale is the inverse of occupancy percentage.
    """

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2:
        raise RuntimeError(f"cannot read a single-channel PGM: {path}")
    values = np.rint((254.0 - image.astype(np.float32)) * (100.0 / 254.0)).astype(
        np.int16
    )
    values[image == 205] = -1
    return values


def _synthetic_map(height: int, width: int) -> np.ndarray:
    """Make a deterministic 0.1-m grid with rooms, doors, and mapped fringe."""

    values = np.full((height, width), -1, dtype=np.int16)
    margin = max(24, min(height, width) // 40)
    values[margin : height - margin, margin : width - margin] = 0
    values[margin : margin + 4, margin : width - margin] = 100
    values[height - margin - 4 : height - margin, margin : width - margin] = 100
    values[margin : height - margin, margin : margin + 4] = 100
    values[margin : height - margin, width - margin - 4 : width - margin] = 100

    # Room walls with 1.0-m door gaps.  The geometry is intentionally broad
    # enough that the core-room mask has multiple meaningful components.
    door_half = max(5, min(height, width) // 150)
    for x in range(width // 4, width - margin, width // 4):
        values[margin : height - margin, x : x + 3] = 100
        center = height // 2 + ((x // 13) % 5 - 2) * (height // 16)
        values[max(margin, center - door_half) : min(height - margin, center + door_half), x : x + 3] = 0
    for y in range(height // 3, height - margin, height // 3):
        values[y : y + 3, margin : width - margin] = 100
        center = width // 2 + ((y // 17) % 5 - 2) * (width // 16)
        values[y : y + 3, max(margin, center - door_half) : min(width - margin, center + door_half)] = 0

    # Small mapped obstacles exercise occupied-component handling.
    for y in range(margin + 80, height - margin - 80, max(80, height // 11)):
        for x in range(margin + 80, width - margin - 80, max(90, width // 10)):
            values[y : y + 8, x : x + 8] = 100
    return values


def _new_segmenter() -> RoomSegmenter:
    """Mirror semantic_mapping_py_pkg/config/default.yaml room defaults."""

    return RoomSegmenter(
        room_free_threshold=20,
        room_unknown_id=-1,
        room_min_component_cells=25,
        room_boundary_margin_cells=1,
        room_core_min_component_cells=40,
        room_core_clearance_cells=7,
        room_small_obstacle_max_cells=0,
        room_remove_enclosed_occupied=True,
        room_enclosed_occupied_max_cells=700,
        room_enclosed_occupied_max_aspect=2.5,
        room_enclosed_occupied_known_ring_ratio=0.95,
        room_enclosed_occupied_free_ring_ratio=0.45,
        room_fill_enclosed_obstacles=False,
        room_portal_cut_enabled=True,
        room_portal_cut_margin_m=0.15,
        room_portal_cut_thickness_cells=2,
        room_portal_detector_min_confirmations=3,
        room_portal_detector_max_center_jump_m=0.4,
        room_portal_hint_merge_distance_m=0.6,
        room_portal_min_width_m=0.5,
        room_portal_max_width_m=2.5,
        room_id_overlap_ratio=0.25,
        room_merge_confirmations=3,
        room_grid_stability_frames=3,
        room_portal_small_component_confidence=70,
    )


def _percentile(samples: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in samples)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * float(percentile) / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _time_ms(func: Callable[[], object], repeats: int) -> tuple[list[float], object]:
    samples: list[float] = []
    result: object = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = func()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return samples, result


def _summary(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "samples_ms": [round(value, 3) for value in samples],
        "mean_ms": round(statistics.fmean(samples), 3),
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(_percentile(samples, 95.0), 3),
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
    }


def _core_mask(values: np.ndarray, clearance_cells: int) -> np.ndarray:
    free = ((values >= 0) & (values <= 20)).astype(np.uint8)
    distance = cv2.distanceTransform((free * 255).astype(np.uint8), cv2.DIST_L2, 5)
    return ((free > 0) & (distance >= float(clearance_cells))).astype(np.uint8)


def _ccl_default(mask: np.ndarray):
    return cv2.connectedComponentsWithStats(mask, 8)


def _ccl_algorithm(mask: np.ndarray, algorithm: int):
    return cv2.connectedComponentsWithStatsWithAlgorithm(
        mask, 8, cv2.CV_32S, algorithm
    )


def _seed_assignment_loop(labels: np.ndarray, stats: np.ndarray, min_area: int) -> np.ndarray:
    """The current Python component extraction + seed-id assignment shape."""

    height, width = labels.shape
    room_ids = np.full(height * width, -1, dtype=np.int32)
    next_temp_id = 1
    component_cells: dict[int, list[int]] = {}
    for component_id in range(1, int(stats.shape[0])):
        if int(stats[component_id, cv2.CC_STAT_AREA]) < min_area:
            continue
        ys, xs = np.where(labels == component_id)
        component_cells[next_temp_id] = [
            int(y) * width + int(x) for y, x in zip(ys.tolist(), xs.tolist())
        ]
        next_temp_id += 1
    for temp_room_id, component in component_cells.items():
        for index in component:
            room_ids[index] = temp_room_id
    return room_ids.reshape(height, width)


def _seed_assignment_vectorised(
    labels: np.ndarray, stats: np.ndarray, min_area: int
) -> np.ndarray:
    """Exact seed-id equivalent of _seed_assignment_loop, without Python cells."""

    accepted = stats[:, cv2.CC_STAT_AREA] >= int(min_area)
    accepted[0] = False
    lookup = np.zeros(stats.shape[0], dtype=np.int32)
    lookup[np.flatnonzero(accepted)] = np.arange(
        1, int(np.count_nonzero(accepted)) + 1, dtype=np.int32
    )
    assigned = lookup[labels]
    return np.where(assigned > 0, assigned, -1).astype(np.int32, copy=False)


def _profile_baseline(grid: SimpleNamespace) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    _new_segmenter().segment(grid)
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(35)
    return stream.getvalue()


def _topology_summary(room_ids: list[int]) -> dict[str, int]:
    array = np.asarray(room_ids, dtype=np.int32)
    return {
        "assigned_cells": int(np.count_nonzero(array >= 0)),
        "room_count": int(np.unique(array[array >= 0]).size),
    }


def _write_report(path: Path, summary: dict) -> None:
    baseline = summary["baseline_segment"]
    seed_loop = summary["seed_assignment_loop"]
    seed_vector = summary["seed_assignment_vectorised"]
    ccl_rows = summary["core_ccl"]
    lines = [
        "# Room segmentation offline benchmark",
        "",
        f"- Source: `{summary['input']['source']}`",
        f"- Grid: {summary['input']['width']} × {summary['input']['height']} "
        f"({summary['input']['cells']:,} cells at {summary['input']['resolution_m']} m/cell)",
        f"- Occupancy: free {summary['input']['free_cells']:,}, occupied "
        f"{summary['input']['occupied_cells']:,}, unknown {summary['input']['unknown_cells']:,}",
        f"- OpenCV: {summary['opencv']['version']}; OpenCV threads: "
        f"{summary['opencv']['threads']}",
        "",
        "## End-to-end current implementation",
        "",
        f"`RoomSegmenter.segment()` median **{baseline['median_ms']:.3f} ms**, "
        f"mean {baseline['mean_ms']:.3f} ms, p95 {baseline['p95_ms']:.3f} ms. "
        f"Output: {baseline['assigned_cells']:,} assigned cells across "
        f"{baseline['room_count']} rooms.",
        "",
        "## Exact core-seed assignment microbenchmark",
        "",
        "This isolates the current `np.where`/Python list-per-component/write-loop "
        "from the equivalent lookup-table assignment. It does not replace the "
        "subsequent topology-aware wavefront yet.",
        "",
        "| implementation | median ms | mean ms | p95 ms | seed output exact |",
        "|---|---:|---:|---:|---:|",
        f"| current Python extraction/write | {seed_loop['median_ms']:.3f} | "
        f"{seed_loop['mean_ms']:.3f} | {seed_loop['p95_ms']:.3f} | yes |",
        f"| vectorised lookup table | {seed_vector['median_ms']:.3f} | "
        f"{seed_vector['mean_ms']:.3f} | {seed_vector['p95_ms']:.3f} | "
        f"{str(summary['seed_assignment_exact']).lower()} |",
        "",
        "## OpenCV core-mask connected-components",
        "",
        "All candidates process the same core mask. The area multiset check "
        "compares topology, independent of arbitrary component label numbering.",
        "",
        "| backend | median ms | mean ms | p95 ms | components | area multiset matches default |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in ccl_rows.items():
        lines.append(
            f"| {name} | {row['median_ms']:.3f} | {row['mean_ms']:.3f} | "
            f"{row['p95_ms']:.3f} | {row['component_count']} | "
            f"{str(row['area_multiset_matches_default']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The full current `segment()` timing includes occupancy conversion, "
            "enclosed-obstacle handling, distance transform, CCL, stable-ID "
            "remap, Python multi-source wavefront, and list stabilisation. The "
            "microbenchmarks show which low-risk operations are worth moving into "
            "the asynchronous room worker first. Results are host-specific; rerun "
            "this script after changing OpenCV thread configuration or map scale.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-pgm",
        type=Path,
        default=DEFAULT_RAW_PGM,
        help="ROS map_saver PGM; falls back to a synthetic same-scale map if absent.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--synthetic-width", type=int, default=1984)
    parser.add_argument("--synthetic-height", type=int, default=1984)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "room_segmentation_benchmark_20260802",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    source = "synthetic"
    if args.input_pgm.exists():
        values = _load_ros_pgm(args.input_pgm)
        source = str(args.input_pgm)
    else:
        values = _synthetic_map(args.synthetic_height, args.synthetic_width)

    grid = _grid(values)
    core = _core_mask(values, clearance_cells=7)
    core_default = _ccl_default(core)
    _count, labels, stats, _centroids = core_default

    baseline_samples, baseline_result = _time_ms(
        lambda: _new_segmenter().segment(grid), args.repeats
    )
    room_ids, _room_conf = baseline_result
    baseline = _summary(baseline_samples)
    baseline.update(_topology_summary(room_ids))

    seed_loop_samples, loop_assignment = _time_ms(
        lambda: _seed_assignment_loop(labels, stats, 40), args.repeats
    )
    seed_vector_samples, vector_assignment = _time_ms(
        lambda: _seed_assignment_vectorised(labels, stats, 40), args.repeats
    )

    algorithms = {"default": None}
    for name in ("CCL_WU", "CCL_GRANA", "CCL_BBDT", "CCL_SAUF", "CCL_SPAGHETTI"):
        if hasattr(cv2, name) and hasattr(cv2, "connectedComponentsWithStatsWithAlgorithm"):
            algorithms[name.removeprefix("CCL_").lower()] = int(getattr(cv2, name))
    ccl_rows: dict[str, dict] = {}
    default_areas = np.sort(stats[1:, cv2.CC_STAT_AREA]).astype(np.int64)
    for name, algorithm in algorithms.items():
        if algorithm is None:
            samples, result = _time_ms(lambda: _ccl_default(core), args.repeats)
        else:
            samples, result = _time_ms(
                lambda algorithm=algorithm: _ccl_algorithm(core, algorithm), args.repeats
            )
        count, _labels, candidate_stats, _centroids = result
        row = _summary(samples)
        row.update(
            {
                "component_count": int(count - 1),
                "area_multiset_matches_default": bool(
                    np.array_equal(
                        default_areas,
                        np.sort(candidate_stats[1:, cv2.CC_STAT_AREA]).astype(np.int64),
                    )
                ),
            }
        )
        ccl_rows[name] = row

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / f"run_{created_at}"
    run_dir.mkdir(parents=True, exist_ok=False)
    profile_text = _profile_baseline(grid)
    (run_dir / "baseline_cprofile.txt").write_text(profile_text, encoding="utf-8")

    summary = {
        "created_at_utc": created_at,
        "input": {
            "source": source,
            "width": int(values.shape[1]),
            "height": int(values.shape[0]),
            "cells": int(values.size),
            "resolution_m": 0.1,
            "free_cells": int(np.count_nonzero((values >= 0) & (values <= 20))),
            "occupied_cells": int(np.count_nonzero(values > 20)),
            "unknown_cells": int(np.count_nonzero(values < 0)),
            "core_cells": int(np.count_nonzero(core)),
        },
        "opencv": {"version": cv2.__version__, "threads": int(cv2.getNumThreads())},
        "repeats": int(args.repeats),
        "baseline_segment": baseline,
        "seed_assignment_loop": _summary(seed_loop_samples),
        "seed_assignment_vectorised": _summary(seed_vector_samples),
        "seed_assignment_exact": bool(np.array_equal(loop_assignment, vector_assignment)),
        "core_ccl": ccl_rows,
    }
    (run_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(run_dir / "benchmark_report.md", summary)
    print(run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
