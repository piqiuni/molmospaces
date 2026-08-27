"""Configurable label filtering for model-produced object detections."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def normalize_detection_label(value):
    return str(value or "").strip().lower().replace(" ", "_")


def load_detection_filter_config(path):
    """Load either a complete semantic-mapping YAML or a filter-only YAML."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "object_detection" in data:
        data = data.get("object_detection") or {}
    if "detection_filter" in data:
        data = data.get("detection_filter") or {}
    if not isinstance(data, dict):
        raise ValueError("detection filter config must be a mapping")
    return data


class DetectionFilter:
    """Filter and canonicalize detector output without touching GT observations."""

    def __init__(self, config=None):
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.excluded_raw_labels = {
            normalize_detection_label(value)
            for value in config.get("excluded_raw_labels", [])
            if normalize_detection_label(value)
        }
        self.excluded_semantic_labels = {
            normalize_detection_label(value)
            for value in config.get("excluded_semantic_labels", [])
            if normalize_detection_label(value)
        }
        self.aliases = {
            normalize_detection_label(source): normalize_detection_label(target)
            for source, target in (config.get("aliases", {}) or {}).items()
            if normalize_detection_label(source) and normalize_detection_label(target)
        }

    def apply(self, detections):
        if not self.enabled:
            return list(detections or [])
        return [filtered for det in (detections or []) if (filtered := self.apply_one(det)) is not None]

    def apply_one(self, detection):
        item = deepcopy(detection)
        semantic = normalize_detection_label(
            item.get("semantic_class")
            or item.get("class")
            or item.get("semantic_name")
            or item.get("label")
        )
        raw = normalize_detection_label(
            item.get("semantic_class_raw")
            or item.get("raw_label")
            or item.get("label")
            or semantic
        )
        if raw in self.excluded_raw_labels:
            return None
        semantic = self.aliases.get(raw, semantic)
        if semantic in self.excluded_semantic_labels:
            return None
        if raw:
            item["semantic_class_raw"] = raw
        if semantic:
            item["semantic_class"] = semantic
        return item
