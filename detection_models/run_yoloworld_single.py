#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import cv2


def add_repo_to_path(repo_dir: Path) -> None:
    repo_dir = repo_dir.resolve()
    sys.path.insert(0, str(repo_dir))
    third_party = repo_dir / "third_party" / "mmyolo"
    if third_party.exists():
        sys.path.insert(0, str(third_party))


def parse_args() -> argparse.Namespace:
    root = Path("/home/user/ldl/molmospaces/detection_models")
    repo_dir = root / "YOLO-World"
    sample_dir = root / "tum_rgbd_scribble_samples"
    parser = argparse.ArgumentParser(description="Run YOLO-World on a single image.")
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=repo_dir,
        help="Path to the YOLO-World repository.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_dir / "configs/pretrain/yolo_world_v2_m_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py",
        help="Model config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_dir / "checkpoints/m_stage2-9987dcb1.pth",
        help="Checkpoint file.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=sample_dir / "kitchen_22_image.png",
        help="Input image path.",
    )
    parser.add_argument(
        "--text",
        default="bed,pillow,nightstand,dresser,wardrobe,chair,table,lamp",
        help="Comma-separated open-vocabulary prompts.",
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "yoloworld_single",
        help="Directory for JSON and annotated image output.",
    )
    return parser.parse_args()


def build_test_pipeline(cfg):
    from mmengine.dataset import Compose
    from mmdet.utils import get_test_pipeline_cfg

    pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
    return Compose(pipeline_cfg)


def run_inference(model, image_path: Path, texts, test_pipeline, score_thr: float, max_dets: int):
    import numpy as np
    import torch

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image_rgb = image_bgr[:, :, [2, 1, 0]]
    data_info = dict(img=image_rgb, img_id=0, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info["inputs"].unsqueeze(0), data_samples=[data_info["data_samples"]])

    with torch.no_grad():
        output = model.test_step(data_batch)[0]
    pred_instances = output.pred_instances
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]
    pred_instances = pred_instances.cpu().numpy()

    results = []
    for bbox, label, score in zip(pred_instances["bboxes"], pred_instances["labels"], pred_instances["scores"]):
        label_idx = int(label)
        results.append(
            {
                "semantic_class": texts[label_idx][0],
                "confidence": float(score),
                "bbox": [float(v) for v in bbox.tolist()],
            }
        )
    return results, image_bgr


def annotate_and_save(image_bgr, detections, output_image: Path) -> None:
    canvas = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        label = f'{det["semantic_class"]} {det["confidence"]:.2f}'
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
    cv2.imwrite(str(output_image), canvas)


def main() -> int:
    args = parse_args()
    add_repo_to_path(args.repo_dir)
    cache_root = args.output_dir / ".hf_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))

    import torch
    import transformers

    if hasattr(torch.optim, "Adafactor") and hasattr(transformers, "Adafactor"):
        delattr(transformers, "Adafactor")

    from mmengine.config import Config
    from mmdet.apis import init_detector

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(str(args.config))
    cfg.load_from = str(args.checkpoint)
    cfg.work_dir = str(args.output_dir / "work_dir")
    model = init_detector(cfg, checkpoint=str(args.checkpoint), device=args.device)

    texts = [[token.strip()] for token in args.text.split(",") if token.strip()] + [[" "]]
    model.reparameterize(texts)
    test_pipeline = build_test_pipeline(cfg)
    detections, image_bgr = run_inference(model, args.image, texts, test_pipeline, args.threshold, args.topk)

    output_json = args.output_dir / f"{args.image.stem}_detections.json"
    output_image = args.output_dir / f"{args.image.stem}_annotated.png"
    output_json.write_text(
        json.dumps(
            {
                "image": str(args.image),
                "config": str(args.config),
                "checkpoint": str(args.checkpoint),
                "device": args.device,
                "detections": detections,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    annotate_and_save(image_bgr, detections, output_image)

    print(f"image: {args.image}")
    print(f"detections: {len(detections)}")
    print(f"json: {output_json}")
    print(f"annotated_image: {output_image}")
    for idx, det in enumerate(detections[:10]):
        print(f"[{idx}] {det['semantic_class']} score={det['confidence']:.3f} bbox={det['bbox']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
