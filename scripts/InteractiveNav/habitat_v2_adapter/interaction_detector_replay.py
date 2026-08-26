#!/usr/bin/env python3
"""Offline replay for open-vocabulary interaction-object detectors.

The script is intentionally independent from Habitat control.  It writes
annotated box/mask frames and a JSONL audit log, so different detector
environments can be compared on exactly the same saved RGB sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np


DEFAULT_PROMPTS = (
    "door",
    "interior door",
    "sliding door",
    "refrigerator",
    "refrigerator door",
    "cabinet",
    "cabinet door",
    "drawer",
    "drawer cabinet",
    "wardrobe",
    "closet",
)


def _color(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return tuple(int(80 + value % 176) for value in digest[:3])


def _label_box(image: np.ndarray, box: list[float], label: str, score: float) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
    color = _color(label)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    text = f"{label} {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    top = max(0, y1 - th - 6)
    cv2.rectangle(image, (x1, top), (min(width - 1, x1 + tw + 4), y1), color, -1)
    cv2.putText(image, text, (x1 + 2, max(th + 1, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def _panel_header(image: np.ndarray, title: str, count: int, latency_ms: float) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (18, 18, 18), -1)
    cv2.putText(
        canvas,
        f"{title} | n={count} | {latency_ms:.1f} ms",
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _frame_paths(directory: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    return [path for path in sorted(directory.iterdir()) if path.suffix.casefold() in suffixes]


def _run_yolo(args: argparse.Namespace, paths: list[Path], prompts: list[str]) -> tuple[list[dict], dict]:
    import torch
    from ultralytics import YOLOE

    model = YOLOE(str(args.weight))
    if args.backend == "yoloe-text":
        model.set_classes(prompts)
    # Warm up without including startup/JIT cost in per-frame latency.
    warm = cv2.imread(str(paths[0]))
    for _ in range(args.warmup):
        model.predict(warm, device=args.device, imgsz=args.imgsz, conf=args.conf,
                      max_det=args.max_det, retina_masks=True, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    rows: list[dict] = []
    latencies: list[float] = []
    box_dir = args.output_dir / "box"
    seg_dir = args.output_dir / "seg"
    box_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not read {path}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = model.predict(
            image, device=args.device, imgsz=args.imgsz, conf=args.conf,
            max_det=args.max_det, retina_masks=True, verbose=False,
        )[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)

        boxes = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        scores = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.empty((0,))
        classes = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes is not None else np.empty((0,), dtype=int)
        names = result.names
        masks = None
        if result.masks is not None:
            masks = result.masks.data.detach().cpu().numpy()
        box_image = image.copy()
        seg_image = image.copy()
        overlay = seg_image.copy()
        detections: list[dict] = []
        for det_index, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
            label = str(names.get(int(cls_id), cls_id))
            coords = [float(value) for value in box]
            mask_area = 0
            if masks is not None and det_index < len(masks):
                mask = masks[det_index]
                if mask.shape != image.shape[:2]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                selected = mask > 0.5
                mask_area = int(selected.sum())
                overlay[selected] = _color(label)
            detections.append({
                "label": label,
                "confidence": float(score),
                "bbox_xyxy": coords,
                "mask_area_px": mask_area,
            })
            _label_box(box_image, coords, label, float(score))
            _label_box(seg_image, coords, label, float(score))
        if masks is not None:
            seg_image = cv2.addWeighted(overlay, 0.45, seg_image, 0.55, 0.0)
            for detection in detections:
                _label_box(seg_image, detection["bbox_xyxy"], detection["label"], detection["confidence"])
        title = "YOLOE-26l PF BOX" if args.backend == "yoloe-pf" else "YOLOE-26l TEXT BOX"
        box_image = _panel_header(box_image, title, len(detections), latency_ms)
        seg_title = "YOLOE-26l PF MASK" if args.backend == "yoloe-pf" else "YOLOE-26l TEXT MASK"
        seg_image = _panel_header(seg_image, seg_title, sum(item["mask_area_px"] > 0 for item in detections), latency_ms)
        cv2.imwrite(str(box_dir / f"frame_{index:04d}.jpg"), box_image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        cv2.imwrite(str(seg_dir / f"frame_{index:04d}.jpg"), seg_image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        rows.append({"index": index, "source": str(path), "latency_ms": latency_ms, "detections": detections})

    peak_allocated = peak_reserved = 0.0
    if torch.cuda.is_available():
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    return rows, {
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_mean_ms": statistics.mean(latencies),
        "peak_vram_allocated_mib": peak_allocated,
        "peak_vram_reserved_mib": peak_reserved,
    }


def _run_grounding_dino(args: argparse.Namespace, paths: list[Path], prompts: list[str]) -> tuple[list[dict], dict]:
    import torch
    from PIL import Image
    import groundingdino.datasets.transforms as transforms
    from groundingdino.util.inference import load_model, predict

    model = load_model(str(args.config), str(args.weight)).to(args.device)
    caption = " . ".join(prompts) + " ."
    warm = cv2.cvtColor(cv2.imread(str(paths[0])), cv2.COLOR_BGR2RGB)
    preprocess = transforms.Compose([
        transforms.RandomResize([800], max_size=1333),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def infer(rgb: np.ndarray):
        tensor, _ = preprocess(Image.fromarray(rgb), None)
        return predict(model=model, image=tensor, caption=caption,
                       box_threshold=args.conf, text_threshold=args.text_threshold,
                       device=args.device)

    for _ in range(args.warmup):
        infer(warm)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    rows: list[dict] = []
    latencies: list[float] = []
    box_dir = args.output_dir / "box"
    box_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, 1):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not read {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        boxes, logits, phrases = infer(rgb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)
        height, width = bgr.shape[:2]
        image = bgr.copy()
        detections: list[dict] = []
        for box, score, phrase in zip(boxes.detach().cpu().numpy(), logits.detach().cpu().numpy(), phrases):
            cx, cy, bw, bh = [float(value) for value in box]
            coords = [
                (cx - bw / 2.0) * width,
                (cy - bh / 2.0) * height,
                (cx + bw / 2.0) * width,
                (cy + bh / 2.0) * height,
            ]
            detection = {"label": str(phrase), "confidence": float(score), "bbox_xyxy": coords}
            detections.append(detection)
            _label_box(image, coords, str(phrase), float(score))
        image = _panel_header(image, "GROUNDING DINO BOX", len(detections), latency_ms)
        cv2.imwrite(str(box_dir / f"frame_{index:04d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        rows.append({"index": index, "source": str(path), "latency_ms": latency_ms, "detections": detections})
    peak_allocated = peak_reserved = 0.0
    if torch.cuda.is_available():
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    return rows, {
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_mean_ms": statistics.mean(latencies),
        "peak_vram_allocated_mib": peak_allocated,
        "peak_vram_reserved_mib": peak_reserved,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("yoloe-pf", "yoloe-text", "grounding-dino"), required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--prompts", default=",".join(DEFAULT_PROMPTS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = _frame_paths(args.frames_dir)
    if not paths:
        raise SystemExit(f"no replay images in {args.frames_dir}")
    prompts = [item.strip() for item in args.prompts.split(",") if item.strip()]
    if args.backend == "grounding-dino" and args.config is None:
        raise SystemExit("--config is required for grounding-dino")

    started = time.perf_counter()
    if args.backend.startswith("yoloe"):
        rows, metrics = _run_yolo(args, paths, prompts)
    else:
        rows, metrics = _run_grounding_dino(args, paths, prompts)
    elapsed = time.perf_counter() - started
    labels: dict[str, int] = {}
    for row in rows:
        for detection in row["detections"]:
            label = detection["label"]
            labels[label] = labels.get(label, 0) + 1
    weight_sha256 = hashlib.sha256(args.weight.read_bytes()).hexdigest()
    summary = {
        "backend": args.backend,
        "weight": str(args.weight),
        "weight_sha256": weight_sha256,
        "frames_dir": str(args.frames_dir),
        "frames": len(rows),
        "confidence_threshold": args.conf,
        "text_threshold": args.text_threshold if args.backend == "grounding-dino" else None,
        "prompts": prompts if args.backend != "yoloe-pf" else None,
        "total_detections": sum(labels.values()),
        "frames_with_detections": sum(bool(row["detections"]) for row in rows),
        "unique_labels": len(labels),
        "label_counts": dict(sorted(labels.items(), key=lambda item: (-item[1], item[0]))),
        "elapsed_s": elapsed,
        **metrics,
    }
    with (args.output_dir / "detections.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
