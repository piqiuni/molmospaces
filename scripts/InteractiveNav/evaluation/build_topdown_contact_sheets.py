#!/usr/bin/env python3
"""Build compact visual indexes from native NavToObj top-down PNG reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


GROUPS = {
    "official_success": {"columns": 4, "title": "Official successes"},
    "official_failure": {"columns": 4, "title": "Official failures"},
    "unevaluable_asset_or_sampling": {"columns": 2, "title": "Asset / sampling exclusions"},
}


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _task_step_summary(debug_dir: Path) -> dict:
    """Recover exact completed task steps from the contiguous simulator frames."""

    manifest_path = debug_dir.parent / "sim_step_frames" / "manifest.jsonl"
    indices: list[int] = []
    if manifest_path.is_file():
        try:
            for line in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
                record = json.loads(line)
                if isinstance(record, dict) and isinstance(record.get("step_index"), int):
                    indices.append(record["step_index"])
        except (OSError, json.JSONDecodeError):
            indices = []
    unique = sorted(set(indices))
    last_step_index = unique[-1] if unique else None
    task_step_count = None if last_step_index is None else last_step_index + 1
    target_selection = _read_json(debug_dir / "target_selection.json")
    horizon = target_selection.get("horizon", {})
    horizon = horizon if isinstance(horizon, dict) else {}
    effective_horizon = horizon.get("effective_task_horizon_steps")
    return {
        "sim_step_manifest": str(manifest_path) if manifest_path.is_file() else None,
        "last_task_step_index": last_step_index,
        "task_step_count": task_step_count,
        "sim_step_frame_count": len(indices),
        "sim_step_indices_contiguous": bool(unique) and unique == list(range(last_step_index + 1)),
        "effective_task_horizon_steps": effective_horizon,
        "task_step_horizon_fraction": (
            None
            if not isinstance(task_step_count, int)
            or not isinstance(effective_horizon, int)
            or effective_horizon <= 0
            else task_step_count / effective_horizon
        ),
    }


def _episode_label(path: Path, outcome: str) -> str:
    prefix = "episode_"
    stem = path.stem
    episode = stem[len(prefix) :].removesuffix(f"_{outcome}") if stem.startswith(prefix) else stem
    return f"episode {int(episode):08d}  |  {outcome.replace('_', ' ')}"


def _font(size: int):
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def build_sheet(
    images: list[Path],
    *,
    output_path: Path,
    outcome: str,
    columns: int,
    thumbnail_width: int,
) -> None:
    from PIL import Image, ImageDraw

    if not images:
        return
    title_font = _font(26)
    label_font = _font(16)
    padding = 12
    label_height = 28
    title_height = 46
    thumbnail_height = int(round(thumbnail_width * 0.64))
    cell_width = thumbnail_width + 2 * padding
    cell_height = thumbnail_height + label_height + 3 * padding
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * cell_width, title_height + rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 10), f"{GROUPS[outcome]['title']} ({len(images)})", fill="#111827", font=title_font)
    for index, path in enumerate(images):
        column = index % columns
        row = index // columns
        x0 = column * cell_width + padding
        y0 = title_height + row * cell_height + padding
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((thumbnail_width, thumbnail_height))
            x = x0 + (thumbnail_width - image.width) // 2
            y = y0 + (thumbnail_height - image.height) // 2
            canvas.paste(image, (x, y))
        draw.rectangle(
            (x0 - 1, y0 - 1, x0 + thumbnail_width, y0 + thumbnail_height),
            outline="#94a3b8",
            width=1,
        )
        draw.text((x0, y0 + thumbnail_height + 6), _episode_label(path, outcome), fill="#111827", font=label_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def build_index(output_dir: Path) -> None:
    """Write one machine-readable index for all rendered outcomes."""

    records: list[dict] = []
    for outcome in GROUPS:
        for image_path in sorted(output_dir.glob(f"episode_*_{outcome}.png")):
            metadata_path = image_path.with_suffix(".json")
            metadata = _read_json(metadata_path)
            adapter_path = Path(str(metadata.get("episode_result", "")))
            adapter = _read_json(adapter_path) if adapter_path.is_file() else {}
            debug_dir = Path(str(metadata.get("debug_dir", "")))
            batch_result_path = debug_dir.parent / "batch_result.json" if debug_dir.is_dir() else Path()
            batch_result = _read_json(batch_result_path) if batch_result_path.is_file() else {}
            step_summary = _task_step_summary(debug_dir) if debug_dir.is_dir() else {}
            result = adapter.get("result", {}) if isinstance(adapter.get("result"), dict) else {}
            trace = adapter.get("trace", []) if isinstance(adapter.get("trace"), list) else []
            terminal_pose = None
            if trace and isinstance(trace[-1], dict):
                base = trace[-1].get("base", {})
                if isinstance(base, dict):
                    terminal_pose = base.get("base_pose_xyyaw")
            records.append(
                {
                    "episode_index": metadata.get("episode_index"),
                    "outcome": outcome,
                    "image": str(image_path),
                    "metadata": str(metadata_path),
                    "official_success": result.get("official_success"),
                    "terminal_reason": result.get("terminal_reason"),
                    "batch_status": batch_result.get("status"),
                    "batch_error_message": batch_result.get("error_message"),
                    "task_description": result.get("task_description"),
                    "distance_m": result.get("distance_m"),
                    "head_camera_visible": result.get("head_camera_visible"),
                    "trajectory_source": metadata.get("trajectory_source"),
                    "trajectory_samples": metadata.get("trajectory_samples"),
                    "terminal_pose_xyyaw": terminal_pose,
                    "eligible_target_total": metadata.get("nav_to_obj_candidate_total"),
                    "eligible_target_observed": len(metadata.get("nav_to_obj_candidates", [])),
                    **step_summary,
                }
            )
    records.sort(key=lambda record: (str(record["outcome"]), int(record["episode_index"] or -1)))
    payload = {
        "schema_version": "native_nav_to_obj_topdown_report_index_v1",
        "total_count": len(records),
        "outcome_counts": {outcome: sum(row["outcome"] == outcome for row in records) for outcome in GROUPS},
        "records": records,
    }
    (output_dir / "topdown_report_index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory containing episode_*.png reports")
    parser.add_argument("--thumbnail-width", type=int, default=420, help="Width of each thumbnail (default: 420)")
    args = parser.parse_args()
    if args.thumbnail_width < 160:
        raise ValueError("--thumbnail-width must be >= 160")
    output_dir = args.output_dir.expanduser().resolve()
    for outcome, settings in GROUPS.items():
        images = sorted(output_dir.glob(f"episode_*_{outcome}.png"))
        if images:
            build_sheet(
                images,
                output_path=output_dir / f"contact_sheet_{outcome}.png",
                outcome=outcome,
                columns=settings["columns"],
                thumbnail_width=args.thumbnail_width,
            )
    build_index(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
