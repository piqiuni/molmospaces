from __future__ import annotations

from pathlib import Path
from typing import Any

from molmo_spaces.molmo_spaces_constants import get_scenes

from .config import SourceConfig


def selected_house_indices(config: SourceConfig) -> list[int]:
    scenes, _version = get_scenes(
        config.scene_dataset, config.data_split, return_version=True
    )
    split_scenes = scenes[config.data_split]
    if config.houses == "all":
        indices = sorted(int(index) for index in split_scenes)
    else:
        indices = sorted({int(index) for index in config.houses})
    end_house = config.end_house
    indices = [
        index
        for index in indices
        if index >= config.start_house and (end_house is None or index < end_house)
    ]
    if config.max_houses is not None:
        indices = indices[: config.max_houses]
    return indices


def build_scene_manifest(config: SourceConfig) -> dict[str, Any]:
    scenes, version = get_scenes(
        config.scene_dataset, config.data_split, return_version=True
    )
    split_scenes = scenes[config.data_split]
    rows = []
    missing = []
    for house_index in selected_house_indices(config):
        variants = split_scenes.get(house_index)
        path = variants.get(config.variant) if isinstance(variants, dict) else variants
        if not path:
            missing.append(house_index)
            if config.missing_scene_policy == "fail":
                raise FileNotFoundError(
                    f"Missing {config.variant} scene for {config.scene_dataset}/"
                    f"{config.data_split} house {house_index}"
                )
            continue
        rows.append(
            {
                "scene_dataset": config.scene_dataset,
                "data_split": config.data_split,
                "house_index": house_index,
                "variant": config.variant,
                "scene_path": str(Path(path)),
                "asset_version": version,
            }
        )
    return {
        "schema_version": "interactive_nav_scene_manifest_v1",
        "scene_dataset": config.scene_dataset,
        "data_split": config.data_split,
        "asset_version": version,
        "variant": config.variant,
        "requested_house_count": len(selected_house_indices(config)),
        "available_house_count": len(rows),
        "missing_house_indices": missing,
        "houses": rows,
    }
