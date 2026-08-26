#!/usr/bin/env python3
"""Aggregate isolated one-scene ObjectNav-v2 worker outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    return parser.parse_args()


def _latest_run(worker_dir: Path) -> Path | None:
    candidates = sorted(path for path in worker_dir.glob("run-*") if path.is_dir())
    return candidates[-1] if candidates else None


def _full_ros_evidence(run_dir: Path) -> dict[str, Any]:
    trace_path = run_dir / "public_policy_trace.jsonl"
    metric_path = run_dir / "posthoc_official_metrics.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    offers = [
        int(row.get("step", 0))
        for row in events
        if row.get("event") == "replan"
        and any(
            str(candidate.get("behavior_type") or "").upper() == "NAVIGATE"
            and str(candidate.get("candidate_id") or "").startswith("target:")
            for candidate in (row.get("candidates") or [])
            if isinstance(candidate, dict)
        )
    ]
    selections = [
        int(row.get("step", 0))
        for row in events
        if row.get("event") == "module2_selection"
        and str(row.get("behavior_type") or "").upper() == "NAVIGATE"
        and str(row.get("candidate_id") or "").startswith("target:")
    ]
    metrics = [json.loads(line) for line in metric_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    distance_by_step = {
        int(row.get("step", 0)): float(row["distance_to_goal"])
        for row in metrics
        if row.get("distance_to_goal") is not None
    }
    selected_distance = None
    min_after = None
    if selections and distance_by_step:
        first = selections[0]
        nearest = min(distance_by_step, key=lambda step: abs(step - first))
        selected_distance = distance_by_step[nearest]
        later = [distance for step, distance in distance_by_step.items() if step >= first]
        min_after = min(later) if later else None
    return {
        "target_perceived": bool(offers),
        "target_detection_hits": len(offers),
        "full_ros_target_candidate_offers": len(offers),
        "module2_target_navigation": bool(selections),
        "module2_target_selection_steps": selections,
        "distance_at_first_target_selection_m": selected_distance,
        "minimum_distance_after_target_selection_m": min_after,
        "approached_after_module2_selection": bool(
            selected_distance is not None and min_after is not None and min_after <= selected_distance - 0.10
        ),
    }


def aggregate(run_root: Path) -> dict[str, Any]:
    worker_root = run_root / "workers"
    rows: list[dict[str, Any]] = []
    images: list[tuple[str, Path]] = []
    failures: list[dict[str, Any]] = []
    episode_lines: list[str] = []

    for worker_dir in sorted(path for path in worker_root.iterdir() if path.is_dir()):
        scene_id = worker_dir.name
        run_dir = _latest_run(worker_dir)
        if run_dir is None or not (run_dir / "posthoc_verdict.json").is_file():
            failures.append({"scene_id": scene_id, "reason": "missing completed run/posthoc verdict"})
            continue
        verdict = json.loads((run_dir / "posthoc_verdict.json").read_text(encoding="utf-8"))
        full_ros = _full_ros_evidence(run_dir)
        if full_ros["target_perceived"] or full_ros["module2_target_navigation"]:
            verdict.update(full_ros)
            (run_dir / "posthoc_verdict.json").write_text(
                json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        episode = json.loads((run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
        row = {
            **verdict,
            "worker_run_dir": str(run_dir),
            "steps": int(episode.get("steps", 0)),
            "spl": float(episode.get("spl", 0.0)),
            "soft_spl": float(episode.get("soft_spl", 0.0)),
            "module2_decisions": int(episode.get("mllm_decisions", 0)),
            "module2_model_selected": int(episode.get("mllm_model_selected", 0)),
            "module1_requests": int(
                episode.get("full_ros_queries", episode.get("module1_detector_queries", 0))
            ),
            "module1_failures": int(
                episode.get("full_ros_failed", episode.get("module1_detector_failed", 0))
            ),
            "navigation_only": bool(summary.get("navigation_only")),
            "module3_enabled": bool(summary.get("module3_enabled")),
        }
        rows.append(row)
        episode_lines.append(json.dumps(row, sort_keys=True) + "\n")
        image_path = run_dir / "topdown_with_targets.png"
        if image_path.is_file():
            images.append((scene_id, image_path))

    aggregate_summary = {
        "protocol": "Habitat Challenge 2023 ObjectNav-v2 / HM3D-Sem v0.2 val / Stretch continuous control",
        "requested_workers": len([path for path in worker_root.iterdir() if path.is_dir()]),
        "completed_workers": len(rows),
        "failed_workers": len(failures),
        "failures": failures,
        "episodes": len(rows),
        "successes": sum(bool(row["official_success"]) for row in rows),
        "success_rate": sum(bool(row["official_success"]) for row in rows) / max(1, len(rows)),
        "target_perceived_scenes": sum(bool(row["target_perceived"]) for row in rows),
        "target_track_promoted_scenes": sum(bool(row["target_track_promoted"]) for row in rows),
        "module2_target_navigation_scenes": sum(bool(row["module2_target_navigation"]) for row in rows),
        "approached_after_module2_selection_scenes": sum(
            bool(row["approached_after_module2_selection"]) for row in rows
        ),
        "module3_enabled": any(bool(row["module3_enabled"]) for row in rows),
        "gt_usage": "target centers/viewpoints and official distances are evaluator-only posthoc diagnostics",
        "scenes": rows,
    }
    (run_root / "aggregate_summary.json").write_text(
        json.dumps(aggregate_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_root / "scene_results.jsonl").write_text("".join(episode_lines), encoding="utf-8")

    if images:
        tile_size = (960, 540)
        columns = 2
        rows_count = (len(images) + columns - 1) // columns
        sheet = Image.new("RGB", (tile_size[0] * columns, tile_size[1] * rows_count), (245, 245, 245))
        for index, (scene_id, image_path) in enumerate(images):
            image = Image.open(image_path).convert("RGB")
            image.thumbnail(tile_size, Image.Resampling.LANCZOS)
            x = (index % columns) * tile_size[0] + (tile_size[0] - image.width) // 2
            y = (index // columns) * tile_size[1] + (tile_size[1] - image.height) // 2
            sheet.paste(image, (x, y))
            result = next((row for row in rows if Path(str(row.get("worker_run_dir"))).parent.name == scene_id), {})
            banner = (
                f"{scene_id}  DET={'Y' if result.get('target_perceived') else 'N'}  "
                f"M2-TARGET={'Y' if result.get('module2_target_navigation') else 'N'}  "
                f"APPROACH={'Y' if result.get('approached_after_module2_selection') else 'N'}  "
                f"SUCCESS={'Y' if result.get('official_success') else 'N'}"
            )
            draw = ImageDraw.Draw(sheet)
            draw.rectangle((x, y, x + min(image.width, 930), y + 24), fill=(255, 255, 255))
            draw.text((x + 6, y + 5), banner, fill=(20, 20, 20))
        sheet_path = run_root / "topdown_10_scenes.png"
        sheet.save(sheet_path, optimize=True)
        aggregate_summary["topdown_contact_sheet"] = str(sheet_path)
        (run_root / "aggregate_summary.json").write_text(
            json.dumps(aggregate_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return aggregate_summary


def main() -> int:
    args = _parse_args()
    print(json.dumps(aggregate(args.run_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
