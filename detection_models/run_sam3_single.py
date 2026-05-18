#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOM_VOCABULARY = [
    "kitchen",
    "bedroom",
    "bathroom",
    "living room",
    "storage",
    "balcony",
    "door",
]

OBJECT_VOCABULARY = [
    "bed",
    "pillow",
    "nightstand",
    "dresser",
    "wardrobe",
    "chair",
    "table",
    "desk",
    "lamp",
    "sofa",
    "cabinet",
    "drawer",
    "shelf",
    "box",
    "sink",
    "toilet",
    "bathtub",
    "shower",
    "refrigerator",
    "fridge",
    "microwave",
    "oven",
    "stove",
    "door handle",
]


def add_repo_to_path(repo_dir: Path) -> None:
    repo_dir = repo_dir.resolve()
    sys.path.insert(0, str(repo_dir))


def parse_args() -> argparse.Namespace:
    root = Path("/home/user/ldl/molmospaces/detection_models")
    repo_dir = root / "sam3"
    sample_dir = root / "tum_rgbd_scribble_samples"
    parser = argparse.ArgumentParser(description="Run SAM3 on a single image with one text prompt.")
    parser.add_argument("--repo-dir", type=Path, default=repo_dir)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_dir / "checkpoints/facebook/sam3/sam3.pt",
        help="SAM3 checkpoint path.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=sample_dir / "kitchen_22_image.png",
        help="Input image path.",
    )
    parser.add_argument("--prompt", default="", help="Single text prompt for SAM3.")
    parser.add_argument(
        "--prompts",
        default="",
        help="Comma-separated prompts. If set, overrides --prompt and vocabulary presets.",
    )
    parser.add_argument(
        "--vocab",
        choices=["all", "room", "object"],
        default="all",
        help="Default vocabulary preset used when no prompt is provided.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dedup-iou", type=float, default=0.75)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "sam3_single",
        help="Directory for JSON, masks, and overlay output.",
    )
    return parser.parse_args()


def save_mask(mask: np.ndarray, path: Path) -> None:
    mask_u8 = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_u8, mode="L").save(path)


def slugify(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower()).strip("_")
    return value or "prompt"


def resolve_prompts(args: argparse.Namespace):
    if args.prompts.strip():
        return [token.strip() for token in args.prompts.split(",") if token.strip()]
    if args.prompt.strip():
        return [args.prompt.strip()]
    if args.vocab == "room":
        return ROOM_VOCABULARY
    if args.vocab == "object":
        return OBJECT_VOCABULARY
    return ROOM_VOCABULARY + OBJECT_VOCABULARY


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def deduplicate_detections(detections, iou_threshold: float):
    sorted_detections = sorted(
        detections,
        key=lambda det: (float(det["confidence"]), float(det["mask_area"])),
        reverse=True,
    )
    kept = []
    for candidate in sorted_detections:
        duplicate = False
        for existing in kept:
            if mask_iou(candidate["mask"], existing["mask"]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def save_overlay(image: Image.Image, detections, output_path: Path) -> None:
    import cv2

    image_np = np.array(image.convert("RGB"))
    overlay = image_np.copy()
    rng = np.random.default_rng(0)
    for det in detections:
        color = rng.integers(32, 255, size=3, dtype=np.uint8)
        mask = det["mask"]
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(int(v) for v in color.tolist()), 2)
        cv2.putText(
            overlay,
            f'{det["prompt"]} {det["confidence"]:.2f}',
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            tuple(int(v) for v in color.tolist()),
            2,
        )
    Image.fromarray(overlay).save(output_path)


def run_single_prompt(processor, image: Image.Image, prompt: str):
    state = processor.set_image(image, state={})
    state = processor.set_text_prompt(prompt=prompt, state=state)
    boxes = state["boxes"].detach().cpu().numpy()
    scores = state["scores"].detach().cpu().numpy()
    masks = state["masks"].detach().cpu().numpy()
    detections = []
    for box, score, mask in zip(boxes, scores, masks):
        mask_bool = np.asarray(mask).squeeze().astype(bool)
        detections.append(
            {
                "prompt": prompt,
                "confidence": float(score),
                "bbox": [float(v) for v in box.tolist()],
                "mask_area": int(mask_bool.sum()),
                "mask": mask_bool,
            }
        )
    return detections


def main() -> int:
    args = parse_args()
    add_repo_to_path(args.repo_dir)

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    prompts = resolve_prompts(args)
    model_device = "cuda" if str(args.device).startswith("cuda") else str(args.device)
    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        device=model_device,
        eval_mode=True,
    )
    processor = Sam3Processor(model, device=model_device, confidence_threshold=args.threshold)

    all_detections = []
    for prompt in prompts:
        all_detections.extend(run_single_prompt(processor, image, prompt))

    deduped = deduplicate_detections(all_detections, args.dedup_iou)

    serializable = []
    detections_for_overlay = []
    run_slug = slugify(args.prompt or args.prompts or args.vocab)
    for idx, det in enumerate(deduped):
        mask_path = args.output_dir / f"{args.image.stem}_{run_slug}_mask_{idx:02d}.png"
        save_mask(det["mask"], mask_path)
        serializable.append(
            {
                "prompt": det["prompt"],
                "confidence": float(det["confidence"]),
                "bbox": [float(v) for v in det["bbox"]],
                "mask_path": str(mask_path),
                "mask_area": int(det["mask_area"]),
            }
        )
        detections_for_overlay.append(
            {
                "prompt": det["prompt"],
                "confidence": float(det["confidence"]),
                "bbox": [float(v) for v in det["bbox"]],
                "mask": det["mask"],
            }
        )

    overlay_path = args.output_dir / f"{args.image.stem}_{run_slug}_overlay.png"
    save_overlay(image, detections_for_overlay, overlay_path)
    output_json = args.output_dir / f"{args.image.stem}_{run_slug}_detections.json"
    output_json.write_text(
        json.dumps(
            {
                "image": str(args.image),
                "checkpoint": str(args.checkpoint),
                "device": args.device,
                "prompts": prompts,
                "vocab": args.vocab,
                "dedup_iou": args.dedup_iou,
                "raw_detection_count": len(all_detections),
                "detections": serializable,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"image: {args.image}")
    print(f"prompt_count: {len(prompts)}")
    print(f"prompts: {prompts}")
    print(f"raw_detections: {len(all_detections)}")
    print(f"detections: {len(serializable)}")
    print(f"json: {output_json}")
    print(f"overlay: {overlay_path}")
    for idx, det in enumerate(serializable[:10]):
        print(
            f"[{idx}] prompt={det['prompt']} score={det['confidence']:.3f} "
            f"bbox={det['bbox']} mask_area={det['mask_area']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
