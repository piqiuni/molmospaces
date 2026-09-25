#!/usr/bin/env python3
"""Apply conservative, task-oriented post-processing to PF replay results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SEMANTIC_MAPPING_SCRIPTS = (
    REPO_ROOT
    / "Interactive-Nav-SG-nav"
    / "src"
    / "semantic_mapping_py_pkg"
    / "scripts"
)
if str(SEMANTIC_MAPPING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_MAPPING_SCRIPTS))

from semantic_mapping_py_pkg.detection_filter import DetectionFilter, load_detection_filter_config

DEFAULT_DETECTOR_CONFIG = (
    REPO_ROOT
    / "Interactive-Nav-SG-nav"
    / "src"
    / "semantic_mapping_py_pkg"
    / "config"
    / "default.yaml"
)


REGION_LABELS = {
    "hall", "hallway", "corridor", "room", "hospital room", "server room",
    "elevator lobby", "elevator shaft", "hotel lobby", "living space", "gallery",
    "pet store", "suite", "studio", "basement", "courtyard", "jail cell",
    "kitchen floor", "floor", "ceiling", "wall", "darkness", "dark",
    "morning fog", "storm", "stormy", "wide", "close-up", "watermark overlay stamp",
    "assemble", "receive", "share", "grow", "type", "modern", "lesson", "association",
}
REGION_TOKENS = {
    "hall", "hallway", "corridor", "room", "floor", "lobby", "laboratory",
    "gallery", "space", "shop", "store", "ballroom", "classroom", "basement",
    "courtyard", "suite", "studio", "hospital", "weather", "fog", "storm",
}
OBJECT_TOKENS = {
    "door", "cabinet", "drawer", "refrigerator", "fridge", "chair", "bed", "desk",
    "table", "lamp", "shelf", "locker", "mirror", "sink", "toilet", "elevator",
    "window", "box", "bottle", "person", "computer", "camera", "microphone",
}
TASK_OBJECT_LABELS = {
    "door", "elevator door", "cabinet", "side cabinet", "file cabinet", "medicine cabinet",
    "drawer", "drawer cabinet", "refrigerator", "fridge", "wardrobe", "closet", "chair",
    "computer chair", "office chair", "electric chair", "bed", "couch", "sofa", "television",
    "tv", "tv monitor", "monitor", "screen", "lamp", "lamp shade", "desk", "office desk",
    "shelf", "locker", "sink", "toilet", "stove", "microwave", "storage box", "box", "person",
    "table", "workbench", "whiteboard", "mirror", "bathroom mirror", "air conditioning", "dish washer",
    "window screen",
}
ALIAS_GROUPS = {
    "television": {"television", "tv", "tv monitor", "tv_monitor", "monitor", "screen"},
    "couch": {"couch", "sofa", "loveseat"},
    "potted plant": {"potted plant", "potted_plant", "plant", "houseplant", "succulent", "flower"},
    "refrigerator": {"refrigerator", "fridge"},
    "cabinet": {"cabinet", "side cabinet", "file cabinet", "medicine cabinet", "locker", "bureau"},
    "door": {"door", "bathroom door", "interior door", "sliding door", "elevator door", "exit", "exit door"},
    "drawer": {"drawer", "drawer cabinet"},
}
CANONICAL = {label: canonical for canonical, labels in ALIAS_GROUPS.items() for label in labels}


def iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1e-9, area_first + area_second - intersection)


def canonical_label(label: str) -> str:
    normalized = str(label).casefold().replace("_", " ").strip()
    return CANONICAL.get(normalized, normalized)


def is_region_label(raw: str, canonical: str) -> bool:
    if raw in {"exit", "exit door"} or canonical == "door" and raw in {"exit", "exit door"}:
        return False
    if raw in REGION_LABELS or canonical in REGION_LABELS:
        return True
    words = set(raw.split())
    # A label such as "office desk" remains an object; "physics laboratory"
    # and "dance floor" are scene-level predictions and are removed.
    return bool(words & REGION_TOKENS) and not bool(words & OBJECT_TOKENS)


def optimize_detections(
    detections: list[dict],
    width: int,
    height: int,
    threshold: float,
    profile: str = "scene",
    detection_filter: DetectionFilter | None = None,
) -> list[dict]:
    if detection_filter is not None:
        normalized = []
        for detection in detection_filter.apply(detections):
            item = dict(detection)
            item["label"] = item.get("semantic_class", item.get("label", ""))
            item["raw_label"] = item.get("semantic_class_raw", item.get("raw_label", item["label"]))
            normalized.append(item)
        detections = normalized
    candidates = []
    image_area = float(width * height)
    for original in detections:
        confidence = float(original.get("confidence", 0.0))
        if confidence < threshold:
            continue
        raw = str(original.get("raw_label", original.get("label", ""))).casefold().replace("_", " ").strip()
        semantic = str(original.get("label", raw)).casefold().replace("_", " ").strip()
        canonical = canonical_label(semantic)
        if profile == "interaction" and raw not in TASK_OBJECT_LABELS and canonical not in TASK_OBJECT_LABELS:
            continue
        box = [float(value) for value in original["bbox_xyxy"]]
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if is_region_label(raw, canonical):
            continue
        # Region-sized boxes are usually scene descriptions rather than objects.
        if area / image_area > 0.68 and canonical not in {"door", "cabinet", "refrigerator"}:
            continue
        item = dict(original)
        item["raw_label"] = raw
        item["label"] = canonical
        item["postprocess"] = "canonicalized"
        candidates.append(item)
    # Class-aware NMS also merges PF aliases that describe the same instance.
    kept: list[dict] = []
    for item in sorted(candidates, key=lambda value: float(value["confidence"]), reverse=True):
        if any(item["label"] == previous["label"] and iou(item["bbox_xyxy"], previous["bbox_xyxy"]) >= 0.45 for previous in kept):
            continue
        kept.append(item)
    return sorted(kept, key=lambda value: (-float(value["confidence"]), value["label"], value["bbox_xyxy"]))


def draw(image: np.ndarray, detections: list[dict], title: str) -> np.ndarray:
    canvas = image.copy()
    for item in detections:
        box = [int(round(value)) for value in item["bbox_xyxy"]]
        color = (0, 220, 80) if item["label"] in {"door", "cabinet", "drawer", "refrigerator"} else (255, 180, 0)
        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, 3, cv2.LINE_AA)
        text = f'{item["label"]} {float(item["confidence"]):.2f}'
        cv2.putText(canvas, text, (box[0] + 2, max(22, box[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (12, 12, 12), -1)
    cv2.putText(canvas, f"{title} | kept={len(detections)}", (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--source-name", default="YOLOE-26l PF")
    parser.add_argument("--profile", choices=("scene", "interaction"), default="interaction")
    parser.add_argument("--detector-config", type=Path, default=DEFAULT_DETECTOR_CONFIG)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detection_filter = DetectionFilter(load_detection_filter_config(args.detector_config))
    rows = [json.loads(line) for line in args.input_jsonl.read_text().splitlines() if line.strip()]
    optim_rows = []
    stats = {str(value): {"frames_with_boxes": 0, "boxes": 0} for value in (0.25, 0.35, 0.45, 0.55)}
    box_dir = args.output_dir / "box"
    box_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        image = cv2.imread(str(row["source"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(row["source"])
        height, width = image.shape[:2]
        for key in stats:
            detections = optimize_detections(row["detections"], width, height, float(key), args.profile, detection_filter)
            stats[key]["frames_with_boxes"] += int(bool(detections))
            stats[key]["boxes"] += len(detections)
        detections = optimize_detections(row["detections"], width, height, args.threshold, args.profile, detection_filter)
        optim_rows.append({"index": row["index"], "source": row["source"], "latency_ms": row.get("latency_ms", 0.0), "detections": detections})
        cv2.imwrite(str(box_dir / f"frame_{int(row['index']):04d}.jpg"), draw(image, detections, f"{args.source_name} OPT conf>={args.threshold:.2f}"), [cv2.IMWRITE_JPEG_QUALITY, 96])
    labels = {}
    for row in optim_rows:
        for item in row["detections"]:
            labels[item["label"]] = labels.get(item["label"], 0) + 1
    summary = {
        "source": str(args.input_jsonl), "frames": len(rows), "threshold": args.threshold,
        "profile": args.profile,
        "boxes": sum(len(row["detections"]) for row in optim_rows),
        "frames_with_boxes": sum(bool(row["detections"]) for row in optim_rows),
        "label_counts": dict(sorted(labels.items(), key=lambda item: (-item[1], item[0]))),
        "threshold_sweep": stats,
        "rules": {"detector_config": str(args.detector_config), "region_labels_removed": sorted(REGION_LABELS), "class_aware_iou": 0.45, "large_box_area_ratio_removed": 0.68},
    }
    with (args.output_dir / "detections.jsonl").open("w", encoding="utf-8") as handle:
        for row in optim_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
