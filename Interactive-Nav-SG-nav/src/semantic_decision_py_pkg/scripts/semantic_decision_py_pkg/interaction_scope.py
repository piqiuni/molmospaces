"""Configuration-only interaction goal admission, independent of graph storage."""

from dataclasses import dataclass
from collections.abc import Mapping


def semantic_family(value: str) -> str:
    label = str(value or "").strip().casefold().replace(" ", "_").replace("-", "_")
    aliases = {
        "refrigerator": "fridge", "freezer": "fridge",
        "portal": "door", "gate": "door", "sliding_door": "door",
        "cabinet": "drawer_cabinet", "drawer": "drawer_cabinet",
        "dresser": "drawer_cabinet", "chest_of_drawers": "drawer_cabinet",
    }
    return aliases.get(label, label)


def interaction_candidate_family(candidate) -> str:
    get = candidate.get if isinstance(candidate, Mapping) else lambda key, default=None: getattr(candidate, key, default)
    metadata = get("metadata") or {}
    # Public semantic identity takes priority over a stale detector target name.
    for value in (
        metadata.get("interaction_semantic_type"), metadata.get("semantic_name"),
        get("target_name"), (get("interaction_command") or {}).get("container_kind"),
        metadata.get("node_type"),
    ):
        if value:
            return semantic_family(value)
    return ""


@dataclass(frozen=True)
class InteractionGoalScope:
    enabled: bool = True
    allowed_semantic_types: tuple[str, ...] | None = None

    @classmethod
    def from_config(cls, config, *, legacy_types=()):
        if config is None:
            # Empty historical allow-lists meant unrestricted (simulation).
            return cls(allowed_semantic_types=tuple(map(semantic_family, legacy_types)) or None)
        if not isinstance(config, Mapping):
            raise ValueError("interaction_goals must be a mapping")
        enabled = config.get("enabled", True)
        values = config.get("allowed_semantic_types")
        if not isinstance(enabled, bool) or not isinstance(values, (list, tuple)):
            raise ValueError("interaction_goals requires boolean enabled and a list of allowed_semantic_types")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("interaction_goals.allowed_semantic_types must contain nonempty names")
        # An explicit empty list denies all interaction goals; it never widens scope.
        return cls(enabled, tuple(dict.fromkeys(map(semantic_family, values))))

    def allows_label(self, label: str) -> bool:
        return self.enabled and (
            self.allowed_semantic_types is None
            or semantic_family(label) in self.allowed_semantic_types
        )

    def allows_candidate(self, candidate) -> bool:
        behavior = candidate.get("behavior_type") if isinstance(candidate, Mapping) else candidate.behavior_type
        return behavior != "INTERACT" or self.allows_label(interaction_candidate_family(candidate))
