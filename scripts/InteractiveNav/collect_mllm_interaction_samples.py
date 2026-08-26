#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2


INTERACTION_KEYWORDS = (
    "door",
    "doorway",
    "gate",
    "fridge",
    "refrigerator",
    "cabinet",
    "dresser",
    "drawer",
    "wardrobe",
    "closet",
    "cupboard",
)
EXCLUDE_KEYWORDS = ("toilet", "sofa", "bed", "table", "countertop", "shelf", "safe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract labelled interaction-object crops from realtime-GT step manifests."
    )
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-visible-fraction", type=float, default=0.20)
    parser.add_argument("--min-visible-pixels", type=int, default=64)
    parser.add_argument("--min-bbox-area-px", type=int, default=512)
    parser.add_argument("--max-distance-m", type=float, default=6.0)
    parser.add_argument("--crop-margin-ratio", type=float, default=0.10)
    parser.add_argument("--crop-max-side-px", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--max-samples-per-object-state", type=int, default=1)
    parser.add_argument("--include-houses", nargs="*", type=int, default=[])
    return parser.parse_args()


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for raw_line in stream:
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def normalized_text(observation: dict[str, Any]) -> str:
    values = (
        observation.get("semantic_name"),
        observation.get("category"),
        observation.get("name"),
        observation.get("source_object_name"),
        observation.get("asset_id"),
    )
    return " ".join(str(value or "").casefold() for value in values)


def is_interaction_object(observation: dict[str, Any]) -> bool:
    text = normalized_text(observation)
    if any(keyword in text for keyword in EXCLUDE_KEYWORDS):
        return False
    if bool(observation.get("is_door") or observation.get("is_movable_door")):
        return True
    if any(keyword in text for keyword in INTERACTION_KEYWORDS):
        return True
    return bool(observation.get("is_receptacle") and observation.get("is_articulable"))


def interaction_class(observation: dict[str, Any]) -> str:
    return "portal" if bool(observation.get("is_door") or observation.get("is_movable_door")) else "container"


def normalized_joint_open_fraction(joint: dict[str, Any]) -> float | None:
    joint_range = joint.get("joint_range") or []
    if not isinstance(joint_range, (list, tuple)) or len(joint_range) < 2:
        return None
    try:
        lower, upper = float(joint_range[0]), float(joint_range[1])
        value = float(joint.get("joint_value", 0.0))
    except (TypeError, ValueError):
        return None
    span = upper - lower
    if abs(span) < 1e-6:
        return None
    return max(0.0, min(1.0, (value - lower) / span))


def coarse_state(observation: dict[str, Any]) -> str:
    joint = {
        "joint_range": observation.get("joint_range"),
        "joint_value": observation.get("joint_value"),
    }
    fraction = normalized_joint_open_fraction(joint)
    if fraction is None:
        infos = observation.get("joint_infos") or []
        fractions = [normalized_joint_open_fraction(item) for item in infos if isinstance(item, dict)]
        fractions = [value for value in fractions if value is not None]
        if not fractions:
            return "unknown"
        fraction = max(fractions)
    if fraction <= 0.15:
        return "closed"
    if fraction >= 0.67:
        return "open"
    return "ajar"


def interaction_parts(observation: dict[str, Any], state: str) -> list[dict[str, Any]]:
    parts = []
    for index, joint in enumerate(observation.get("joint_infos") or []):
        if not isinstance(joint, dict):
            continue
        fraction = normalized_joint_open_fraction(joint)
        part_state = state
        if fraction is not None:
            part_state = "closed" if fraction <= 0.15 else "open" if fraction >= 0.67 else "ajar"
        parts.append(
            {
                "part_id": str(joint.get("joint_name") or f"part_{index}"),
                "type": str(joint.get("joint_type") or "door"),
                "state": part_state,
                "confidence": 1.0,
            }
        )
    return parts


def valid_observation(observation: dict[str, Any], args: argparse.Namespace) -> bool:
    if not is_interaction_object(observation):
        return False
    visible_fraction = observation.get("visible_fraction")
    if (
        visible_fraction is not None
        and float(visible_fraction or 0.0) < args.min_visible_fraction
    ):
        return False
    if int(observation.get("visible_pixels", 0) or 0) < args.min_visible_pixels:
        return False
    distance_m = float(observation.get("distance_m", 0.0) or 0.0)
    if args.max_distance_m > 0.0 and distance_m > args.max_distance_m:
        return False
    bbox = observation.get("bbox_2d") or observation.get("projected_bbox_2d") or []
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    try:
        area = abs(float(bbox[2]) - float(bbox[0])) * abs(float(bbox[3]) - float(bbox[1]))
    except (TypeError, ValueError):
        return False
    return area >= args.min_bbox_area_px


def sample_score(observation: dict[str, Any]) -> float:
    bbox = observation.get("bbox_2d") or observation.get("projected_bbox_2d") or [0, 0, 0, 0]
    area = abs(float(bbox[2]) - float(bbox[0])) * abs(float(bbox[3]) - float(bbox[1]))
    distance_m = max(0.1, float(observation.get("distance_m", 0.0) or 0.1))
    visible_fraction = max(0.0, float(observation.get("visible_fraction", 0.0) or 0.0))
    return math.log1p(area) * (0.5 + visible_fraction) / distance_m


def crop_image(
    frame_path: Path,
    bbox: list[float],
    output_path: Path,
    margin_ratio: float,
    max_side_px: int,
    jpeg_quality: int,
) -> dict[str, Any] | None:
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    raw_x0, raw_y0, raw_x1, raw_y1 = [int(round(float(value))) for value in bbox[:4]]
    left, right = sorted((raw_x0, raw_x1))
    top, bottom = sorted((raw_y0, raw_y1))
    margin_x = int(round((right - left) * max(0.0, margin_ratio)))
    margin_y = int(round((bottom - top) * max(0.0, margin_ratio)))
    left = max(0, min(width - 1, left - margin_x))
    right = min(width, max(left + 1, right + margin_x))
    top = max(0, min(height - 1, top - margin_y))
    bottom = min(height, max(top + 1, bottom + margin_y))
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    crop_height, crop_width = crop.shape[:2]
    scale = min(1.0, float(max_side_px) / max(crop_width, crop_height))
    if scale < 1.0:
        crop = cv2.resize(
            crop,
            (
                max(1, int(round(crop_width * scale))),
                max(1, int(round(crop_height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]):
        return None
    return {
        "source_size": [width, height],
        "crop_bbox": [left, top, right, bottom],
        "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
        "source_bytes": frame_path.stat().st_size,
        "crop_bytes": output_path.stat().st_size,
    }


def manifest_house_index(manifest_path: Path) -> int | None:
    for parent in (manifest_path.parent, *manifest_path.parents):
        name = parent.name
        if name.startswith("house_"):
            try:
                return int(name.removeprefix("house_"))
            except ValueError:
                return None
    return None


def select_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    transitions: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    previous_states: dict[tuple[str, str, str], str] = {}
    manifest_paths = sorted(
        {
            manifest_path.resolve()
            for input_dir in args.input_dir
            for manifest_path in input_dir.glob("**/sim_step_frames/manifest.jsonl")
        }
    )
    for manifest_path in manifest_paths:
        house_index = manifest_house_index(manifest_path)
        if args.include_houses and house_index not in set(args.include_houses):
            continue
        seen_frame_ids: set[tuple[str, int]] = set()
        for row in read_json_lines(manifest_path):
            gt_payload = row.get("gt_observations") or {}
            if not bool(gt_payload.get("observation_performed", False)):
                continue
            episode_id = str(gt_payload.get("episode_id") or "")
            frame_index = int(gt_payload.get("frame_index", -1) or -1)
            frame_key = (episode_id, frame_index)
            if frame_index >= 0 and frame_key in seen_frame_ids:
                continue
            seen_frame_ids.add(frame_key)
            frame_path = Path(str(row.get("frame") or ""))
            if not frame_path.is_file():
                continue
            for observation in gt_payload.get("observations") or []:
                if not isinstance(observation, dict) or not valid_observation(observation, args):
                    continue
                object_id = str(
                    observation.get("instance_id")
                    or observation.get("source_object_name")
                    or observation.get("object_id")
                    or ""
                )
                if not object_id:
                    continue
                state = coarse_state(observation)
                scene_id = manifest_path.parents[1].name
                key = (scene_id, episode_id, object_id, state)
                record = {
                    "scene_id": scene_id,
                    "house_ind": house_index,
                    "episode_id": episode_id,
                    "step_index": int(row.get("step_index", -1) or -1),
                    "frame_index": frame_index,
                    "frame_path": str(frame_path),
                    "object_id": object_id,
                    "source_object_name": str(observation.get("source_object_name") or ""),
                    "semantic_name": str(observation.get("semantic_name") or observation.get("name") or ""),
                    "category": str(observation.get("category") or ""),
                    "interaction_class_gt": interaction_class(observation),
                    "coarse_state_gt": state,
                    "interaction_parts_gt": interaction_parts(observation, state),
                    "bbox_2d": list(observation.get("bbox_2d") or observation.get("projected_bbox_2d") or []),
                    "visible_fraction": float(observation.get("visible_fraction", 0.0) or 0.0),
                    "visible_pixels": int(observation.get("visible_pixels", 0) or 0),
                    "distance_m": float(observation.get("distance_m", 0.0) or 0.0),
                    "joint_infos": list(observation.get("joint_infos") or []),
                    "score": sample_score(observation),
                }
                current = selected.get(key)
                if current is None or record["score"] > current["score"]:
                    selected[key] = record
                state_key = (scene_id, episode_id, object_id)
                previous_state = previous_states.get(state_key)
                if previous_state and previous_state != state and state in {"open", "closed", "ajar"}:
                    transition_key = (*state_key, previous_state, state)
                    current_transition = transitions.get(transition_key)
                    if current_transition is None or record["score"] > current_transition["score"]:
                        transitions[transition_key] = {**record, "previous_state": previous_state}
                previous_states[state_key] = state
    return list(selected.values()), list(transitions.values())


def materialize_samples(
    records: list[dict[str, Any]],
    output_dir: Path,
    prefix: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    materialized = []
    for index, record in enumerate(sorted(records, key=lambda item: (item["scene_id"], item["object_id"], item["coarse_state_gt"]))):
        filename = f"{prefix}_{index:05d}_{record['scene_id']}_{record['object_id'][:18]}_{record['coarse_state_gt']}.jpg"
        crop_path = output_dir / "crops" / filename
        crop_meta = crop_image(
            Path(record["frame_path"]),
            record["bbox_2d"],
            crop_path,
            args.crop_margin_ratio,
            args.crop_max_side_px,
            args.jpeg_quality,
        )
        if crop_meta is None:
            continue
        materialized.append(
            {
                **record,
                "crop_path": str(crop_path.resolve()),
                "crop": crop_meta,
            }
        )
    return materialized


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.input_dir = [input_dir.expanduser().resolve() for input_dir in args.input_dir]
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attribute_records, verification_records = select_samples(args)
    attribute_records = materialize_samples(attribute_records, args.output_dir, "attribute", args)
    verification_records = materialize_samples(verification_records, args.output_dir, "verification", args)
    write_jsonl(args.output_dir / "attribute_samples.jsonl", attribute_records)
    write_jsonl(args.output_dir / "verification_samples.jsonl", verification_records)
    summary = {
        "input_dirs": [str(input_dir) for input_dir in args.input_dir],
        "attribute_sample_count": len(attribute_records),
        "verification_sample_count": len(verification_records),
        "scenes": sorted({record["scene_id"] for record in attribute_records}),
        "attribute_by_class": {
            label: sum(1 for record in attribute_records if record["interaction_class_gt"] == label)
            for label in sorted({record["interaction_class_gt"] for record in attribute_records})
        },
        "attribute_by_state": {
            label: sum(1 for record in attribute_records if record["coarse_state_gt"] == label)
            for label in sorted({record["coarse_state_gt"] for record in attribute_records})
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
