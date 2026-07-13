#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOE prompt-free segmentation on a single image.")
    parser.add_argument("--model", required=True, help="Path to a local YOLOE prompt-free weight file.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--save-plot", action="store_true", help="Save overlay visualization.")
    parser.add_argument(
        "--output-dir",
        default="/home/user/ldl/molmospaces/detection_models/yoloe/outputs",
        help="Directory to store JSON and visualization outputs.",
    )
    return parser.parse_args()


def mask_area(mask_tensor):
    if mask_tensor is None:
        return None
    mask = mask_tensor.astype(np.uint8)
    return int(mask.sum())


def ensure_path(path_str):
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def main():
    args = parse_args()
    model_path = ensure_path(args.model)
    image_path = ensure_path(args.image)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLOE(str(model_path))
    results = model.predict(
        source=str(image_path),
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=True,
        verbose=True,
        save=False,
    )

    if not results:
        raise RuntimeError("YOLOE returned no results list.")

    result = results[0]
    names = result.names
    boxes = result.boxes
    masks = result.masks
    detections = []

    num_masks = 0 if masks is None or masks.data is None else int(masks.data.shape[0])

    if boxes is not None:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)

        mask_array = None
        if masks is not None and masks.data is not None:
            mask_array = masks.data.detach().cpu().numpy()

        for idx in range(len(xyxy)):
            cls_id = int(classes[idx])
            detection = {
                "index": idx,
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": float(confs[idx]),
                "bbox_xyxy": [float(v) for v in xyxy[idx].tolist()],
                "has_mask": bool(mask_array is not None and idx < len(mask_array)),
                "mask_area": None,
            }
            if mask_array is not None and idx < len(mask_array):
                detection["mask_area"] = mask_area(mask_array[idx] > 0.5)
            detections.append(detection)

    stem = image_path.stem
    model_stem = model_path.stem
    json_path = output_dir / f"{stem}_{model_stem}_detections.json"
    png_path = output_dir / f"{stem}_{model_stem}_overlay.png"

    summary = {
        "image": str(image_path),
        "model": str(model_path),
        "device": args.device,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "num_detections": len(detections),
        "num_masks": num_masks,
        "detections": detections,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.save_plot:
        plotted_bgr = result.plot()
        cv2.imwrite(str(png_path), plotted_bgr)

    print(f"Saved JSON to: {json_path}")
    if args.save_plot:
        print(f"Saved overlay to: {png_path}")
    print(f"Detections: {len(detections)}")
    for det in detections[:10]:
        print(
            f"[{det['index']}] {det['class_name']} "
            f"conf={det['confidence']:.3f} "
            f"bbox={det['bbox_xyxy']} "
            f"mask_area={det['mask_area']}"
        )


if __name__ == "__main__":
    main()
