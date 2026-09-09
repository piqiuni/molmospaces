from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PRIMARY_CATEGORIES = {"fridge", "dresser"}


def distance_bin(distance_m: float) -> str:
    if distance_m < 3.0:
        return "near"
    if distance_m < 6.0:
        return "medium"
    return "far"


def start_distance(start: dict[str, Any]) -> float:
    return float(
        start.get("planar_distance_to_object_m", start["distance_to_object_m"])
    )


def load_candidates(catalog_path: Path) -> dict[int, list[dict[str, Any]]]:
    with open(catalog_path) as handle:
        payload = json.load(handle)
    houses = payload.get("houses", payload) if isinstance(payload, dict) else payload
    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for house in houses:
        house_index = int(house["house_index"])
        for container in house["containers"]:
            container_category = str(container.get("category", "")).lower()
            if container_category not in PRIMARY_CATEGORIES:
                continue
            for target in container.get("strict_contained_objects", []):
                starts = sorted(
                    target.get("source_starts", []),
                    key=lambda row: int(row["episode_index"]),
                )
                if not starts:
                    continue
                candidates[house_index].append(
                    {
                        "house_index": house_index,
                        "container_name": container["name"],
                        "container_category": container.get("category"),
                        "container_asset_id": container.get("asset_id"),
                        "container_aabb_size": container["aabb_size"],
                        "object_name": target["name"],
                        "object_category": target["category"],
                        "object_aabb_size": target["aabb_size"],
                        "source_starts": starts,
                    }
                )
    return dict(candidates)


def choose_start(
    candidate: dict[str, Any], distance_counts: Counter[str]
) -> tuple[dict[str, Any], list[int]]:
    ranked = sorted(
        candidate["source_starts"],
        key=lambda row: (
            distance_counts[distance_bin(start_distance(row))],
            -start_distance(row),
            int(row["episode_index"]),
        ),
    )
    selected = ranked[0]
    preferred = [int(selected["episode_index"])] + [
        int(row["episode_index"])
        for row in ranked[1:]
        if int(row["episode_index"]) != int(selected["episode_index"])
    ]
    return selected, preferred


def candidate_score(
    candidate: dict[str, Any],
    container_counts: Counter[str],
    category_counts: dict[str, Counter[str]],
    asset_counts: Counter[str],
    distance_counts: Counter[str],
    used_categories: set[str],
) -> tuple[Any, ...]:
    container_category = str(candidate["container_category"]).lower()
    object_category = str(candidate["object_category"]).lower()
    best_distance_count = min(
        distance_counts[distance_bin(start_distance(row))]
        for row in candidate["source_starts"]
    )
    return (
        container_counts[container_category],
        category_counts[container_category][object_category],
        int(object_category in used_categories),
        asset_counts[str(candidate.get("container_asset_id"))],
        best_distance_count,
        candidate["container_name"],
        candidate["object_name"],
    )


def backup_score(
    backup: dict[str, Any], primary: dict[str, Any]
) -> tuple[Any, ...]:
    same_container_category = (
        backup["container_category"] == primary["container_category"]
    )
    same_object_category = backup["object_category"] == primary["object_category"]
    same_container = backup["container_name"] == primary["container_name"]
    return (
        -int(same_container_category and same_object_category),
        -int(same_container_category),
        -int(same_container),
        backup["container_name"],
        backup["object_name"],
    )


def strip_candidate(
    candidate: dict[str, Any],
    preferred_source_episode_indices: list[int],
    *,
    rank: int,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key != "source_starts"
    } | {
        "candidate_rank": rank,
        "preferred_source_episode_indices": preferred_source_episode_indices,
    }


def build_dynamic_collection_plan(
    catalog_path: Path,
    *,
    max_samples: int,
    samples_per_house: int,
    target_house_count: int | None = None,
    house_indices: list[int] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Build an in-memory fine-collection plan from the rough catalog.

    The plan fixes the house set but keeps every strict pair in each house as a
    fallback candidate. Fine collection decides path and interaction validity.
    """
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if samples_per_house <= 0:
        raise ValueError("samples_per_house must be positive")

    candidates_by_house = load_candidates(catalog_path)
    rng = random.Random(seed)
    eligible_houses = list(candidates_by_house)
    tie_breakers = {house_index: rng.random() for house_index in eligible_houses}

    def house_priority(house_index: int) -> tuple[Any, ...]:
        rows = candidates_by_house[house_index]
        return (
            -len({str(row["container_category"]).lower() for row in rows}),
            -len({str(row["object_category"]).lower() for row in rows}),
            -len({str(row.get("container_asset_id")) for row in rows}),
            -len(rows),
            tie_breakers[house_index],
            house_index,
        )

    if house_indices is not None:
        house_order = [
            int(house_index)
            for house_index in house_indices
            if int(house_index) in candidates_by_house
        ]
    else:
        house_order = sorted(eligible_houses, key=house_priority)
        if target_house_count is not None:
            house_order = house_order[:target_house_count]

    requested_by_house = {house_index: 0 for house_index in house_order}
    remaining = max_samples
    for _ in range(samples_per_house):
        for house_index in house_order:
            if remaining <= 0:
                break
            requested_by_house[house_index] += 1
            remaining -= 1
        if remaining <= 0:
            break
    house_order = [
        house_index for house_index in house_order if requested_by_house[house_index] > 0
    ]

    container_counts: Counter[str] = Counter()
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    asset_counts: Counter[str] = Counter()
    distance_counts: Counter[str] = Counter()
    primary_by_house: dict[int, list[dict[str, Any]]] = defaultdict(list)
    selected_keys: set[tuple[int, str, str]] = set()

    for sample_index in range(samples_per_house):
        for house_index in house_order:
            if sample_index >= requested_by_house[house_index]:
                continue
            used_categories = {
                str(row["object_category"]).lower()
                for row in primary_by_house[house_index]
            }
            available = [
                row
                for row in candidates_by_house[house_index]
                if (house_index, row["container_name"], row["object_name"])
                not in selected_keys
            ]
            if not available:
                continue
            primary = min(
                available,
                key=lambda row: candidate_score(
                    row,
                    container_counts,
                    category_counts,
                    asset_counts,
                    distance_counts,
                    used_categories,
                ),
            )
            primary_by_house[house_index].append(primary)
            selected_keys.add(
                (house_index, primary["container_name"], primary["object_name"])
            )
            selected_start, _ = choose_start(primary, distance_counts)
            container_category = str(primary["container_category"]).lower()
            object_category = str(primary["object_category"]).lower()
            container_counts[container_category] += 1
            category_counts[container_category][object_category] += 1
            asset_counts[str(primary.get("container_asset_id"))] += 1
            distance_counts[distance_bin(start_distance(selected_start))] += 1

    houses = []
    for house_index in house_order:
        primaries = primary_by_house[house_index]
        primary_keys = {
            (row["container_name"], row["object_name"]) for row in primaries
        }
        used_categories = {
            str(row["object_category"]).lower() for row in primaries
        }
        fallbacks = [
            row
            for row in candidates_by_house[house_index]
            if (row["container_name"], row["object_name"]) not in primary_keys
        ]
        fallbacks.sort(
            key=lambda row: candidate_score(
                row,
                container_counts,
                category_counts,
                asset_counts,
                distance_counts,
                used_categories,
            )
        )
        ordered = primaries + fallbacks
        candidates = []
        for rank, candidate in enumerate(ordered):
            selected_start, preferred = choose_start(candidate, distance_counts)
            enriched = {
                **candidate,
                "selected_start_distance_m": float(start_distance(selected_start)),
                "selected_start_distance_bin": distance_bin(
                    start_distance(selected_start)
                ),
            }
            candidates.append(strip_candidate(enriched, preferred, rank=rank))
        houses.append(
            {
                "house_index": house_index,
                "target_sample_count": requested_by_house[house_index],
                "candidates": candidates,
            }
        )

    return {
        "schema_version": "container_interaction_collection_plan_v2",
        "source_catalog": str(catalog_path),
        "selection": {
            "mode": "fixed",
            "max_samples": max_samples,
            "samples_per_house": samples_per_house,
            "seed": seed,
            "selected_house_count": len(houses),
            "requested_sample_count": sum(
                row["target_sample_count"] for row in houses
            ),
            "eligible_house_count": len(eligible_houses),
            "target_house_count": target_house_count,
            "container_category_counts": dict(container_counts),
            "object_category_counts": {
                key: dict(value) for key, value in category_counts.items()
            },
            "distance_bin_counts": dict(distance_counts),
        },
        "houses": houses,
    }


def select_manifest(args: argparse.Namespace) -> dict[str, Any]:
    candidates_by_house = load_candidates(args.catalog)
    rng = random.Random(args.seed)
    eligible_houses = [
        house_index
        for house_index, rows in candidates_by_house.items()
        if not args.require_full_house_quota or len(rows) >= args.max_per_house
    ]
    tie_breakers = {house_index: rng.random() for house_index in eligible_houses}

    def house_priority(house_index: int) -> tuple[Any, ...]:
        rows = candidates_by_house[house_index]
        container_categories = {
            str(row["container_category"]).lower() for row in rows
        }
        object_categories = {str(row["object_category"]).lower() for row in rows}
        assets = {str(row.get("container_asset_id")) for row in rows}
        return (
            -len(container_categories),
            -len(object_categories),
            -len(assets),
            -len(rows),
            tie_breakers[house_index],
            house_index,
        )

    house_order = sorted(eligible_houses, key=house_priority)
    if args.target_house_count is not None:
        house_order = house_order[: args.target_house_count]
    rng.shuffle(house_order)

    container_counts: Counter[str] = Counter()
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    asset_counts: Counter[str] = Counter()
    distance_counts: Counter[str] = Counter()
    selected_by_house: dict[int, list[dict[str, Any]]] = defaultdict(list)
    selected_keys: set[tuple[int, str, str]] = set()

    for slot_index in range(args.max_per_house):
        for house_index in house_order:
            if sum(map(len, selected_by_house.values())) >= args.max_samples:
                break
            used_categories = {
                row["object_category"] for row in selected_by_house[house_index]
            }
            available = [
                row
                for row in candidates_by_house[house_index]
                if (house_index, row["container_name"], row["object_name"])
                not in selected_keys
            ]
            if not available:
                continue
            primary = min(
                available,
                key=lambda row: candidate_score(
                    row,
                    container_counts,
                    category_counts,
                    asset_counts,
                    distance_counts,
                    used_categories,
                ),
            )
            selected_start, preferred = choose_start(primary, distance_counts)
            primary = {
                **primary,
                "preferred_source_episode_indices": preferred,
                "selected_start_distance_m": float(
                    start_distance(selected_start)
                ),
                "selected_start_distance_bin": distance_bin(
                    start_distance(selected_start)
                ),
            }
            selected_by_house[house_index].append(primary)
            selected_keys.add(
                (house_index, primary["container_name"], primary["object_name"])
            )
            container_category = str(primary["container_category"]).lower()
            object_category = str(primary["object_category"]).lower()
            container_counts[container_category] += 1
            category_counts[container_category][object_category] += 1
            asset_counts[str(primary.get("container_asset_id"))] += 1
            distance_counts[
                distance_bin(start_distance(selected_start))
            ] += 1

    houses = []
    for house_index in sorted(selected_by_house):
        reserved: set[tuple[str, str]] = {
            (row["container_name"], row["object_name"])
            for row in selected_by_house[house_index]
        }
        slots = []
        for slot_index, primary in enumerate(selected_by_house[house_index]):
            pool = [
                row
                for row in candidates_by_house[house_index]
                if (row["container_name"], row["object_name"]) not in reserved
            ]
            pool.sort(key=lambda row: backup_score(row, primary))
            backups = pool[: args.redundancy_per_slot]
            slot_candidates = [
                strip_candidate(
                    primary,
                    primary["preferred_source_episode_indices"],
                    rank=0,
                )
            ]
            for rank, backup in enumerate(backups, start=1):
                backup_start, preferred = choose_start(backup, distance_counts)
                backup = {
                    **backup,
                    "selected_start_distance_m": float(
                        start_distance(backup_start)
                    ),
                    "selected_start_distance_bin": distance_bin(
                        start_distance(backup_start)
                    ),
                }
                slot_candidates.append(strip_candidate(backup, preferred, rank=rank))
                reserved.add((backup["container_name"], backup["object_name"]))
            slots.append(
                {
                    "slot_id": f"house_{house_index}_slot_{slot_index}",
                    "primary_container_category": primary["container_category"],
                    "primary_object_category": primary["object_category"],
                    "candidates": slot_candidates,
                }
            )
        houses.append({"house_index": house_index, "slots": slots})

    return {
        "schema_version": "container_interaction_candidate_manifest_v1",
        "source_catalog": str(args.catalog),
        "selection": {
            "max_samples": args.max_samples,
            "max_per_house": args.max_per_house,
            "redundancy_per_slot": args.redundancy_per_slot,
            "seed": args.seed,
            "selected_slot_count": sum(len(row["slots"]) for row in houses),
            "selected_house_count": len(houses),
            "eligible_house_count": len(eligible_houses),
            "target_house_count": args.target_house_count,
            "require_full_house_quota": args.require_full_house_quota,
            "container_category_counts": dict(container_counts),
            "object_category_counts": {
                key: dict(value) for key, value in category_counts.items()
            },
            "distance_bin_counts": dict(distance_counts),
        },
        "houses": houses,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select balanced container-interaction collection candidates."
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--max_per_house", type=int, default=2)
    parser.add_argument("--redundancy_per_slot", type=int, default=2)
    parser.add_argument("--target_house_count", type=int)
    parser.add_argument(
        "--require_full_house_quota",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = select_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest["selection"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
