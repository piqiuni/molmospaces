#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLOE


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark YOLOE prompt-free models on a small image set.")
    parser.add_argument("--models", nargs="+", required=True, help="One or more local YOLOE weights.")
    parser.add_argument("--images", nargs="+", required=True, help="One or more input images.")
    parser.add_argument("--device", default="cuda:0", help="Inference device.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup rounds per model.")
    parser.add_argument("--loops", type=int, default=5, help="Measured rounds per model.")
    parser.add_argument(
        "--output",
        default="/home/user/ldl/molmospaces/detection_models/yoloe/outputs/benchmark_yoloe_prompt_free.json",
        help="Path to benchmark summary JSON.",
    )
    return parser.parse_args()


def ensure_existing(paths):
    out = []
    for item in paths:
        path = Path(item).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        out.append(path)
    return out


def run_once(model, images, device, imgsz, conf, iou):
    start = time.perf_counter()
    results = model.predict(
        source=[str(p) for p in images],
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        agnostic_nms=True,
        verbose=False,
        save=False,
        stream=False,
    )
    elapsed = time.perf_counter() - start
    det_counts = []
    for result in results:
        det_counts.append(0 if result.boxes is None else len(result.boxes))
    return elapsed, det_counts


def main():
    args = parse_args()
    model_paths = ensure_existing(args.models)
    image_paths = ensure_existing(args.images)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "device": args.device,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "warmup": args.warmup,
        "loops": args.loops,
        "images": [str(p) for p in image_paths],
        "models": [],
    }

    for model_path in model_paths:
        model = YOLOE(str(model_path))

        for _ in range(args.warmup):
            run_once(model, image_paths, args.device, args.imgsz, args.conf, args.iou)

        latencies = []
        detection_hist = []
        for _ in range(args.loops):
            elapsed, det_counts = run_once(model, image_paths, args.device, args.imgsz, args.conf, args.iou)
            latencies.append(elapsed)
            detection_hist.append(det_counts)

        total_images = len(image_paths) * args.loops
        total_time = sum(latencies)
        avg_batch_ms = total_time / args.loops * 1000.0
        avg_image_ms = total_time / total_images * 1000.0
        fps = total_images / total_time if total_time > 0 else 0.0

        model_entry = {
            "model": str(model_path),
            "avg_batch_ms": avg_batch_ms,
            "avg_image_ms": avg_image_ms,
            "fps": fps,
            "latencies_ms": [v * 1000.0 for v in latencies],
            "detection_hist": detection_hist,
        }
        summary["models"].append(model_entry)

        print(f"Model: {model_path.name}")
        print(f"  avg_batch_ms={avg_batch_ms:.2f}")
        print(f"  avg_image_ms={avg_image_ms:.2f}")
        print(f"  fps={fps:.2f}")
        print(f"  detections_per_loop={detection_hist}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved benchmark summary to: {output_path}")


if __name__ == "__main__":
    main()
