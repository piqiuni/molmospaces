from __future__ import annotations


DEFAULT_INTERACTION_KEYWORDS = (
    "door",
    "doorway",
    "gate",
    "barrier",
    "fridge",
    "refrigerator",
    "cabinet",
    "dresser",
    "drawer",
    "wardrobe",
    "closet",
    "cupboard",
)
DEFAULT_EXCLUDE_KEYWORDS = (
    "toilet",
    "sofa",
    "bed",
    "table",
    "countertop",
    "shelf",
    "safe",
)


def normalized_detection_text(detection: dict) -> str:
    values = (
        detection.get("semantic_name"),
        detection.get("category"),
        detection.get("name"),
        detection.get("source_object_name"),
        detection.get("asset_id"),
    )
    return " ".join(str(value or "").casefold() for value in values)


def is_interaction_attribute_candidate(
    detection: dict,
    include_keywords: tuple[str, ...],
    exclude_keywords: tuple[str, ...],
) -> bool:
    text = normalized_detection_text(detection)
    if any(keyword in text for keyword in exclude_keywords):
        return False
    if bool(detection.get("is_door") or detection.get("is_movable_door")):
        return True
    if any(keyword in text for keyword in include_keywords):
        return True
    return bool(detection.get("is_receptacle") and detection.get("is_articulable"))
