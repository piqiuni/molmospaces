import argparse
import csv
import logging
from pathlib import Path

from molmo_spaces.molmo_spaces_constants import get_scenes
from molmo_spaces.utils.scene_maps import ProcTHORMap, iTHORMap


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank navigation scenes by estimated free-space area and optionally export map previews."
    )
    parser.add_argument(
        "--scene_dataset",
        type=str,
        default="procthor-10k",
        help="Scene dataset name, e.g. procthor-10k or ithor",
    )
    parser.add_argument("--data_split", type=str, default="train", help="train/val/test split")
    parser.add_argument("--agent_radius", type=float, default=0.3, help="Robot safety radius in meters")
    parser.add_argument("--px_per_m", type=int, default=100, help="Map resolution in pixels per meter")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="assets/scene_rankings",
        help="Output directory for ranking CSV and optional preview maps",
    )
    parser.add_argument(
        "--save_maps",
        action="store_true",
        help="If set, save occupancy/room-map previews as PNG files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of scenes to process (for quick dry-runs)",
    )
    return parser.parse_args()


def select_map_class(scene_dataset: str):
    dataset = scene_dataset.lower()
    return iTHORMap if "ithor" in dataset else ProcTHORMap


def get_scene_path(variants: dict) -> str | None:
    # Prefer ceiling variants when available as they are commonly used in datagen.
    return variants.get("ceiling") or variants.get("base")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_map = get_scenes(args.scene_dataset, args.data_split)[args.data_split]
    map_cls = select_map_class(args.scene_dataset)
    log.info("Processing %d scenes from %s/%s", len(scene_map), args.scene_dataset, args.data_split)

    rows: list[dict] = []
    processed = 0

    for house_idx, variants in scene_map.items():
        if args.limit is not None and processed >= args.limit:
            break

        scene_path = get_scene_path(variants)
        if scene_path is None:
            continue
        scene_path_obj = Path(scene_path)
        if not scene_path_obj.exists():
            log.warning("Skipping house %s because scene file is missing: %s", house_idx, scene_path)
            continue

        try:
            thormap = map_cls.from_mj_model_path(
                model_path=str(scene_path_obj),
                agent_radius=args.agent_radius,
                px_per_m=args.px_per_m,
                device_id=None,
            )
            occupancy = thormap.occupancy
            h, w = occupancy.shape
            px_per_m = float(thormap.px_per_m)

            bbox_area_m2 = (h / px_per_m) * (w / px_per_m)
            free_area_m2 = float(occupancy.sum()) / (px_per_m**2)
            free_ratio = free_area_m2 / bbox_area_m2 if bbox_area_m2 > 0 else 0.0

            rows.append(
                {
                    "house_idx": int(house_idx),
                    "scene_path": str(scene_path_obj),
                    "resolved_scene_path": str(scene_path_obj.resolve()),
                    "bbox_area_m2": bbox_area_m2,
                    "free_area_m2": free_area_m2,
                    "free_ratio": free_ratio,
                    "map_h_px": int(h),
                    "map_w_px": int(w),
                    "px_per_m": px_per_m,
                }
            )

            if args.save_maps:
                thormap.save(str(out_dir / f"house_{house_idx}_map.png"))

            processed += 1
            if processed % 20 == 0:
                log.info("Processed %d scenes...", processed)
        except Exception as exc:
            log.warning("Skipping house %s due to error: %s", house_idx, exc)

    if not rows:
        raise RuntimeError("No scenes were successfully processed.")

    rows.sort(key=lambda x: x["free_area_m2"], reverse=True)

    csv_path = out_dir / f"{args.scene_dataset}_{args.data_split}_ranking.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    log.info("Ranking written to %s", csv_path)
    top_10 = [row["house_idx"] for row in rows[:10]]
    log.info("Top 10 house indices by free area: %s", top_10)


if __name__ == "__main__":
    main()
