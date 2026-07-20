from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class SourceConfig(BaseModel):
    kind: Literal["scene_split", "nav_benchmark"] = "scene_split"
    scene_dataset: str = "procthor-10k"
    data_split: Literal["train", "val", "test"] = "train"
    benchmark_path: Path | None = None
    houses: Literal["all"] | list[int] = "all"
    start_house: int = 0
    end_house: int | None = None
    max_houses: int | None = None
    variant: Literal["base", "ceiling"] = "base"
    seeds_per_house: int = Field(default=4, ge=1)
    seed_candidate_pool: int = Field(default=4096, ge=64)
    missing_scene_policy: Literal["skip", "fail"] = "skip"
    preferred_object_categories: list[str] = Field(default_factory=list)
    preferred_object_names: list[str] = Field(default_factory=list)
    min_start_goal_distance_m: float = Field(default=0.0, ge=0.0)
    max_start_goal_distance_m: float | None = Field(default=None, gt=0.0)
    prefer_longest_start_goal: bool = True

    @model_validator(mode="after")
    def validate_source(self) -> "SourceConfig":
        if self.max_start_goal_distance_m is not None and (
            self.max_start_goal_distance_m < self.min_start_goal_distance_m
        ):
            raise ValueError(
                "max_start_goal_distance_m must be >= min_start_goal_distance_m"
            )
        return self


class ChannelRecipes(BaseModel):
    single_required_closed: float = Field(default=0.60, ge=0.0)
    distractor_only: float = Field(default=0.20, ge=0.0)
    required_plus_distractor: float = Field(default=0.20, ge=0.0)
    distractor_k_min: int = Field(default=1, ge=1)
    distractor_k_max: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def validate_ratios(self) -> "ChannelRecipes":
        total = (
            self.single_required_closed
            + self.distractor_only
            + self.required_plus_distractor
        )
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"channel recipe ratios must sum to 1.0, got {total}")
        if self.distractor_k_max < self.distractor_k_min:
            raise ValueError("distractor_k_max must be >= distractor_k_min")
        return self


class ContainerRecipes(BaseModel):
    container_hinged_door: float = Field(default=0.50, ge=0.0)
    container_sliding_drawer: float = Field(default=0.50, ge=0.0)
    max_type_ratio: float = Field(default=0.70, gt=0.0, le=1.0)


class MixedRecipes(BaseModel):
    canonical_mixed_required: float = Field(default=0.70, ge=0.0)
    mixed_required_plus_distractor: float = Field(default=0.30, ge=0.0)
    distractor_door_k_min: int = Field(default=1, ge=1)
    distractor_door_k_max: int = Field(default=2, ge=1)
    candidate_types: list[str] = Field(default_factory=lambda: ["mixed_required_verified"])

    @model_validator(mode="after")
    def validate_ratios(self) -> "MixedRecipes":
        total = self.canonical_mixed_required + self.mixed_required_plus_distractor
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"mixed recipe ratios must sum to 1.0, got {total}")
        if self.candidate_types != ["mixed_required_verified"]:
            raise ValueError("formal mixed collection only accepts mixed_required_verified")
        return self


class DomainsConfig(BaseModel):
    channel: ChannelRecipes = Field(default_factory=ChannelRecipes)
    container: ContainerRecipes = Field(default_factory=ContainerRecipes)
    mixed: MixedRecipes = Field(default_factory=MixedRecipes)


class CollectionModeConfig(BaseModel):
    mode: Literal["light", "full"] = "light"
    schema_version: Literal["interactive_nav_v3"] = "interactive_nav_v3"
    open_gt_control: bool = False
    synthetic_wrong_action_rollout: bool = False
    domains: DomainsConfig = Field(default_factory=DomainsConfig)

    @model_validator(mode="after")
    def reject_excluded_data(self) -> "CollectionModeConfig":
        if self.open_gt_control:
            raise ValueError("open_gt_control is excluded from this collection protocol")
        if self.synthetic_wrong_action_rollout:
            raise ValueError("synthetic wrong-action rollouts are excluded from this protocol")
        return self


class BalanceConfig(BaseModel):
    total_samples: int = Field(default=100, ge=3)
    strategy: Literal["strict_valid_episode_count"] = "strict_valid_episode_count"
    domain_order: list[Literal["channel", "container", "mixed"]] = Field(
        default_factory=lambda: ["channel", "container", "mixed"]
    )
    max_samples_per_house: dict[str, int] = Field(
        default_factory=lambda: {"channel": 2, "container": 2, "mixed": 1}
    )

    def target_counts(self) -> dict[str, int]:
        base, remainder = divmod(self.total_samples, len(self.domain_order))
        return {
            domain: base + (1 if index < remainder else 0)
            for index, domain in enumerate(self.domain_order)
        }


class ExecutorConfig(BaseModel):
    executor: str = "force"
    target_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    max_steps: int = Field(default=1500, ge=1)
    tolerance: float = Field(default=1e-3, gt=0.0)


class PolicyConfig(BaseModel):
    navigation_executor: str = "astar"
    channel: ExecutorConfig = Field(default_factory=ExecutorConfig)
    container: ExecutorConfig = Field(default_factory=ExecutorConfig)


class RuntimeConfig(BaseModel):
    workers: int = Field(default=4, ge=1)
    seed: int = 20260720
    resume: bool = True
    mujoco_gl: str = "egl"
    save_images: bool = False
    save_plots: bool = False


class RoughConfig(BaseModel):
    container_catalog: Path | None = None
    mixed_catalog: Path | None = None
    generate_if_missing: bool = True

    @model_validator(mode="after")
    def validate_catalog_names(self) -> "RoughConfig":
        if (
            self.container_catalog is not None
            and self.container_catalog.name != "rough_catalog.json"
        ):
            raise ValueError("container_catalog must point to rough_catalog.json")
        if (
            self.mixed_catalog is not None
            and self.mixed_catalog.name != "mixed_rough_catalog.json"
        ):
            raise ValueError("mixed_catalog must point to mixed_rough_catalog.json")
        return self


class FullRolloutConfig(BaseModel):
    max_episodes: int = Field(default=1, ge=1)
    domains: list[Literal["channel", "container", "mixed"]] = Field(
        default_factory=lambda: ["channel", "container", "mixed"]
    )
    max_steps: int = Field(default=500, ge=1)
    max_base_adjustment_steps: int = Field(default=300, ge=1)
    video_fps: float = Field(default=10.0, gt=0.0)
    required_open_fraction: float = Field(default=0.67, ge=0.0, le=1.0)
    image_width: int = Field(default=320, ge=64)
    image_height: int = Field(default=180, ge=64)
    selection_strategy: Literal["shortest_validated_path", "benchmark_order"] = (
        "shortest_validated_path"
    )


class OutputConfig(BaseModel):
    root: Path


class CollectionConfig(BaseModel):
    source: SourceConfig = Field(default_factory=SourceConfig)
    collection: CollectionModeConfig = Field(default_factory=CollectionModeConfig)
    balance: BalanceConfig = Field(default_factory=BalanceConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    rough: RoughConfig = Field(default_factory=RoughConfig)
    full: FullRolloutConfig = Field(default_factory=FullRolloutConfig)
    output: OutputConfig


def load_collection_config(path: Path) -> CollectionConfig:
    payload = yaml.safe_load(path.read_text())
    config = CollectionConfig.model_validate(payload)
    if not config.output.root.is_absolute():
        config.output.root = (Path.cwd() / config.output.root).resolve()
    if (
        config.source.benchmark_path is not None
        and not config.source.benchmark_path.is_absolute()
    ):
        config.source.benchmark_path = (Path.cwd() / config.source.benchmark_path).resolve()
    for field in ("container_catalog", "mixed_catalog"):
        value = getattr(config.rough, field)
        if value is not None and not value.is_absolute():
            setattr(config.rough, field, (Path.cwd() / value).resolve())
    return config
