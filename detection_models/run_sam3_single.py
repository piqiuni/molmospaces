#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
import time
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
    parser = argparse.ArgumentParser(
        description="Run SAM3 multi-prompt detection with RGB-D 3D lifting and benchmark timing."
    )
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
        help="Input RGB image path.",
    )
    parser.add_argument(
        "--depth",
        type=Path,
        default=None,
        help="Input depth image path. Defaults to the paired *_depth.png next to --image.",
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
        "--resolution",
        type=int,
        default=1008,
        help="Square resize resolution for SAM3 API transform.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Build SAM3 image model with torch.compile enabled.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of timed runs.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs before timing.")
    parser.add_argument("--max-depth-m", type=float, default=8.0)
    parser.add_argument("--point-stride", type=int, default=3)
    parser.add_argument("--min-valid-points", type=int, default=24)
    parser.add_argument("--trim-ratio", type=float, default=0.10)
    parser.add_argument(
        "--fx",
        type=float,
        default=525.0,
        help="Camera fx in pixels. Defaults to a TUM RGB-D style pinhole model.",
    )
    parser.add_argument("--fy", type=float, default=525.0, help="Camera fy in pixels.")
    parser.add_argument("--cx", type=float, default=319.5, help="Camera cx in pixels.")
    parser.add_argument("--cy", type=float, default=239.5, help="Camera cy in pixels.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "sam3_single",
        help="Directory for JSON, masks, and overlay output.",
    )
    parser.add_argument(
        "--save-run-artifacts",
        action="store_true",
        help="Save per-run masks and overlays for every repeat instead of only the final run.",
    )
    return parser.parse_args()


def resolve_depth_path(image_path: Path, depth_path: Path | None) -> Path:
    if depth_path is not None:
        return depth_path
    stem = image_path.stem
    if stem.endswith("_image"):
        paired = image_path.with_name(stem[:-6] + "_depth" + image_path.suffix)
        if paired.exists():
            return paired
    raise FileNotFoundError(f"Could not infer depth path for {image_path}")


def save_mask(mask: np.ndarray, path: Path) -> None:
    mask_u8 = mask.astype(np.uint8) * 255
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


def load_depth_meters(depth_path: Path) -> np.ndarray:
    depth = np.asarray(Image.open(depth_path), dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if valid.any() and float(depth[valid].mean()) > 20.0:
        depth = depth / 1000.0
    return depth


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


def pixels_to_camera_points(pixels: np.ndarray, depths: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    cols = pixels[:, 0].astype(np.float32)
    rows = pixels[:, 1].astype(np.float32)
    z = depths.astype(np.float32)
    x = (cols - args.cx) * z / max(args.fx, 1e-6)
    y = (rows - args.cy) * z / max(args.fy, 1e-6)
    return np.stack([x, y, z], axis=1)


def trim_points(points: np.ndarray, trim_ratio: float) -> np.ndarray:
    if points.shape[0] < 4:
        return points
    trim_ratio = min(max(float(trim_ratio), 0.0), 0.45)
    if trim_ratio <= 0.0:
        return points
    mins = np.quantile(points, trim_ratio, axis=0)
    maxs = np.quantile(points, 1.0 - trim_ratio, axis=0)
    keep = np.all((points >= mins) & (points <= maxs), axis=1)
    trimmed = points[keep]
    return trimmed if trimmed.shape[0] >= 4 else points


def mask_to_points(mask: np.ndarray, depth_m: np.ndarray, args: argparse.Namespace):
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None, None
    stride = max(1, int(args.point_stride))
    rows = rows[::stride]
    cols = cols[::stride]
    depths = depth_m[rows, cols]
    valid = np.isfinite(depths) & (depths > 0.0) & (depths <= float(args.max_depth_m))
    if not valid.any():
        return None, None
    pixels = np.stack([cols[valid], rows[valid]], axis=1)
    values = depths[valid]
    return pixels, values


def enrich_detection_3d(detection, depth_m: np.ndarray, args: argparse.Namespace):
    pixels, depths = mask_to_points(detection["mask"], depth_m, args)
    if pixels is None or depths is None or depths.size < args.min_valid_points:
        return None
    points = pixels_to_camera_points(pixels, depths, args)
    points = trim_points(points, args.trim_ratio)
    if points.shape[0] < args.min_valid_points:
        return None
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    size = np.maximum(maxs - mins, 0.0)
    detection = dict(detection)
    detection["point_count"] = int(points.shape[0])
    detection["depth_stats"] = {
        "min_m": float(depths.min()),
        "median_m": float(np.median(depths)),
        "max_m": float(depths.max()),
    }
    detection["visible_box3d_center"] = {
        "x": float(center[0]),
        "y": float(center[1]),
        "z": float(center[2]),
    }
    detection["visible_box3d_size"] = {
        "x": float(size[0]),
        "y": float(size[1]),
        "z": float(size[2]),
    }
    detection["position"] = dict(detection["visible_box3d_center"])
    detection["projection_method"] = "sam3_mask_depth_box3d"
    return detection


def create_empty_datapoint():
    from sam3.train.data.sam3_image_dataset import Datapoint

    return Datapoint(find_queries=[], images=[])


def set_image_for_datapoint(datapoint, pil_image: Image.Image):
    from sam3.train.data.sam3_image_dataset import Image as SAMImage

    width, height = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[height, width])]


def add_text_prompt(datapoint, text_query: str, query_id: int):
    from sam3.train.data.sam3_image_dataset import FindQueryLoaded, InferenceMetadata

    width, height = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=query_id,
                original_image_id=query_id,
                original_category_id=1,
                original_size=[width, height],
                object_id=0,
                frame_index=0,
            ),
        )
    )


def build_api_transform(resolution: int):
    from sam3.train.transforms.basic_for_api import ComposeAPI, NormalizeAPI, RandomResizeAPI, ToTensorAPI

    return ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=int(resolution),
                max_size=int(resolution),
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def run_batched_prompts(model, transform, postprocessor, image: Image.Image, prompts, model_device: str):
    import torch
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api as collate

    datapoint = create_empty_datapoint()
    set_image_for_datapoint(datapoint, image)
    prompt_ids = {}
    for idx, prompt in enumerate(prompts, start=1):
        prompt_ids[idx] = prompt
        add_text_prompt(datapoint, prompt, query_id=idx)

    datapoint = transform(datapoint)
    batch = collate([datapoint], dict_key="dummy")["dummy"]
    batch = copy_data_to_device(batch, torch.device(model_device), non_blocking=True)
    with torch.inference_mode():
        output = model(batch)
        processed_results = postprocessor.process_results(output, batch.find_metadatas)

    detections = []
    for query_id, prompt in prompt_ids.items():
        result = processed_results.get(query_id)
        if not result:
            continue
        boxes = result["boxes"]
        scores = result["scores"]
        masks = result["masks"]
        boxes_np = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
        scores_np = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
        if hasattr(masks, "detach"):
            masks_np = masks.detach().cpu().numpy()
        else:
            masks_np = np.asarray(masks)
        for box, score, mask in zip(boxes_np, scores_np, masks_np):
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


def serializable_detection(det, mask_path: Path):
    return {
        "prompt": det["prompt"],
        "confidence": float(det["confidence"]),
        "bbox": [float(v) for v in det["bbox"]],
        "mask_path": str(mask_path),
        "mask_area": int(det["mask_area"]),
        "point_count": int(det.get("point_count", 0)),
        "depth_stats": det.get("depth_stats"),
        "position": det.get("position"),
        "visible_box3d_center": det.get("visible_box3d_center"),
        "visible_box3d_size": det.get("visible_box3d_size"),
        "projection_method": det.get("projection_method", "sam3_mask_depth_box3d"),
    }


def process_once(model, transform, postprocessor, image, depth_m, prompts, args, model_device):
    import torch

    timings = {
        "prompt_seconds": {},
    }
    if model_device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_total = time.perf_counter()
    start_prompt_batch = time.perf_counter()
    all_detections = run_batched_prompts(model, transform, postprocessor, image, prompts, model_device)
    prompt_batch_time = time.perf_counter() - start_prompt_batch
    per_prompt = prompt_batch_time / max(1, len(prompts))
    for prompt in prompts:
        timings["prompt_seconds"][prompt] = per_prompt
    timings["prompt_batch_seconds"] = prompt_batch_time

    deduped = deduplicate_detections(all_detections, args.dedup_iou)
    start_3d = time.perf_counter()
    enriched = []
    for det in deduped:
        det3d = enrich_detection_3d(det, depth_m, args)
        if det3d is not None:
            enriched.append(det3d)
    timings["lift_3d_seconds"] = time.perf_counter() - start_3d
    timings["total_seconds"] = time.perf_counter() - start_total
    timings["raw_detection_count"] = len(all_detections)
    timings["dedup_detection_count"] = len(deduped)
    timings["final_detection_count"] = len(enriched)
    if model_device == "cuda":
        timings["peak_cuda_memory_allocated_mb"] = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        timings["peak_cuda_memory_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0))
    return enriched, timings


def save_outputs(image, detections, prompts, args, run_slug, run_index=None):
    run_dir = args.output_dir
    if run_index is not None:
        run_dir = args.output_dir / f"run_{run_index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
    serializable = []
    overlay_input = []
    for idx, det in enumerate(detections):
        mask_path = run_dir / f"{args.image.stem}_{run_slug}_mask_{idx:02d}.png"
        save_mask(det["mask"], mask_path)
        serializable.append(serializable_detection(det, mask_path))
        overlay_input.append(
            {
                "prompt": det["prompt"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "mask": det["mask"],
            }
        )

    overlay_path = run_dir / f"{args.image.stem}_{run_slug}_overlay.png"
    save_overlay(image, overlay_input, overlay_path)
    return serializable, overlay_path


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    args = parse_args()
    add_repo_to_path(args.repo_dir)

    from sam3 import build_sam3_image_model
    from sam3.eval.postprocessors import PostProcessImage
    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    depth_path = resolve_depth_path(args.image, args.depth)
    image = Image.open(args.image).convert("RGB")
    depth_m = load_depth_meters(depth_path)
    prompts = resolve_prompts(args)
    model_device = "cuda" if str(args.device).startswith("cuda") else str(args.device)

    if model_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script is configured to run on GPU.")

    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        device=model_device,
        eval_mode=True,
        compile=args.compile,
    )
    transform = build_api_transform(args.resolution)
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=args.threshold,
        to_cpu=False,
    )

    if model_device == "cuda":
        torch.cuda.synchronize()
    for _ in range(max(0, int(args.warmup_runs))):
        process_once(model, transform, postprocessor, image, depth_m, prompts, args, model_device)
        if model_device == "cuda":
            torch.cuda.synchronize()

    run_records = []
    final_detections = []
    for run_idx in range(max(1, int(args.repeats))):
        if model_device == "cuda":
            torch.cuda.synchronize()
        detections, timings = process_once(model, transform, postprocessor, image, depth_m, prompts, args, model_device)
        if model_device == "cuda":
            torch.cuda.synchronize()
        run_records.append(timings)
        final_detections = detections
        if args.save_run_artifacts:
            save_outputs(image, detections, prompts, args, slugify(args.prompt or args.prompts or args.vocab), run_idx)

    run_slug = slugify(args.prompt or args.prompts or args.vocab)
    serializable, overlay_path = save_outputs(image, final_detections, prompts, args, run_slug)

    totals = [record["total_seconds"] for record in run_records]
    lift_times = [record["lift_3d_seconds"] for record in run_records]
    peak_allocated = [record.get("peak_cuda_memory_allocated_mb", 0.0) for record in run_records]
    peak_reserved = [record.get("peak_cuda_memory_reserved_mb", 0.0) for record in run_records]
    benchmark = {
        "repeats": len(run_records),
        "warmup_runs": int(args.warmup_runs),
        "total_seconds": {
            "mean": float(np.mean(totals)),
            "std": float(np.std(totals)),
            "min": float(np.min(totals)),
            "max": float(np.max(totals)),
            "p50": percentile(totals, 50),
            "p90": percentile(totals, 90),
        },
        "lift_3d_seconds": {
            "mean": float(np.mean(lift_times)),
            "std": float(np.std(lift_times)),
            "min": float(np.min(lift_times)),
            "max": float(np.max(lift_times)),
            "p50": percentile(lift_times, 50),
            "p90": percentile(lift_times, 90),
        },
        "peak_cuda_memory_allocated_mb": {
            "mean": float(np.mean(peak_allocated)),
            "std": float(np.std(peak_allocated)),
            "min": float(np.min(peak_allocated)),
            "max": float(np.max(peak_allocated)),
            "p50": percentile(peak_allocated, 50),
            "p90": percentile(peak_allocated, 90),
        },
        "peak_cuda_memory_reserved_mb": {
            "mean": float(np.mean(peak_reserved)),
            "std": float(np.std(peak_reserved)),
            "min": float(np.min(peak_reserved)),
            "max": float(np.max(peak_reserved)),
            "p50": percentile(peak_reserved, 50),
            "p90": percentile(peak_reserved, 90),
        },
        "runs": run_records,
    }

    output_json = args.output_dir / f"{args.image.stem}_{run_slug}_detections.json"
    output_json.write_text(
        json.dumps(
            {
                "image": str(args.image),
                "depth": str(depth_path),
                "checkpoint": str(args.checkpoint),
                "device": args.device,
                "compile": bool(args.compile),
                "prompts": prompts,
                "vocab": args.vocab,
                "dedup_iou": args.dedup_iou,
                "camera_model": {
                    "fx": args.fx,
                    "fy": args.fy,
                    "cx": args.cx,
                    "cy": args.cy,
                },
                "resolution": int(args.resolution),
                "depth_units": "meters",
                "benchmark": benchmark,
                "detections": serializable,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mean_total = benchmark["total_seconds"]["mean"]
    fps = 1.0 / mean_total if mean_total > 0.0 else math.inf
    print(f"image: {args.image}")
    print(f"depth: {depth_path}")
    print(f"prompts: {prompts}")
    print(f"repeats: {len(run_records)} warmup: {args.warmup_runs}")
    print(
        "timing_total_seconds: "
        f"mean={benchmark['total_seconds']['mean']:.4f} "
        f"std={benchmark['total_seconds']['std']:.4f} "
        f"min={benchmark['total_seconds']['min']:.4f} "
        f"max={benchmark['total_seconds']['max']:.4f} "
        f"fps={fps:.3f}"
    )
    print(
        "timing_3d_seconds: "
        f"mean={benchmark['lift_3d_seconds']['mean']:.4f} "
        f"std={benchmark['lift_3d_seconds']['std']:.4f}"
    )
    print(
        "peak_cuda_memory_mb: "
        f"allocated_mean={benchmark['peak_cuda_memory_allocated_mb']['mean']:.1f} "
        f"allocated_max={benchmark['peak_cuda_memory_allocated_mb']['max']:.1f} "
        f"reserved_mean={benchmark['peak_cuda_memory_reserved_mb']['mean']:.1f} "
        f"reserved_max={benchmark['peak_cuda_memory_reserved_mb']['max']:.1f}"
    )
    print(f"detections: {len(serializable)}")
    print(f"json: {output_json}")
    print(f"overlay: {overlay_path}")
    for idx, det in enumerate(serializable[:10]):
        center = det.get("visible_box3d_center") or {}
        size = det.get("visible_box3d_size") or {}
        print(
            f"[{idx}] prompt={det['prompt']} score={det['confidence']:.3f} "
            f"mask_area={det['mask_area']} points={det['point_count']} "
            f"center=({center.get('x', 0.0):.3f},{center.get('y', 0.0):.3f},{center.get('z', 0.0):.3f}) "
            f"size=({size.get('x', 0.0):.3f},{size.get('y', 0.0):.3f},{size.get('z', 0.0):.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
