"""Runtime scene filtering for the InteractiveNav benchmark.

The frozen benchmark identifies the goal by an exact object instance while the
language instruction usually exposes only its category.  This module applies
the evaluator's controlled ``single-instance`` variant without modifying the
benchmark JSON on disk.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Iterable


# Objaverse/MolmoSpaces names use the asset category as the first token of the
# instance name (for example ``compactdisk_<asset>``), while the benchmark
# language uses a shorter natural-language label (``cd``).  Keep this mapping
# local so the runtime filter does not import an analysis script.
_CATEGORY_ALIASES = {
    "crapper": "toilet",
    "ashcan": "garbagecan",
    "atomizer": "spraybottle",
    "alarmclock": "alarmclock",
    "irishpotato": "potato",
    "compactdisk": "cd",
    "cellulartelephone": "cellphone",
    "barsoap": "soapbar",
}


def _canonical_category(value: Any) -> str:
    token = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    return _CATEGORY_ALIASES.get(token, token)


def _instance_category(instance_name: str) -> str:
    basename = str(instance_name).rsplit("/", 1)[-1]
    return _canonical_category(basename.split("_", 1)[0])


def _same_instance_name(left: str, right: str) -> bool:
    left = str(left)
    right = str(right)
    return (
        left == right
        or left.endswith("/" + right)
        or right.endswith("/" + left)
        or left.rsplit("/", 1)[-1] == right.rsplit("/", 1)[-1]
    )


def apply_same_category_distractor_filter(
    episode_spec: Any,
    interactive_nav: Mapping[str, Any],
    *,
    enabled: bool = True,
    candidate_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Remove non-target instances of the target category from one episode.

    The operation mutates only the per-episode ``EpisodeSpec`` held by the
    evaluator.  The frozen benchmark JSON remains unchanged.  Removal is
    expressed through ``scene_modifications.removed_objects`` so it happens
    before MuJoCo compiles the scene.
    """

    target = interactive_nav.get("target") or {}
    selected_instance = str(
        target.get("selected_instance")
        or episode_spec.task.get("pickup_obj_name", "")
        or ""
    )
    language = getattr(episode_spec, "language", None)
    language_label = (
        (getattr(language, "referral_expressions", None) or {}).get("object_name")
        if language is not None
        else None
    )
    target_category = _canonical_category(target.get("category") or language_label)
    object_poses = episode_spec.scene_modifications.object_poses
    pose_names = {str(name) for name in object_poses}
    candidate_names = (
        {str(name) for name in candidate_names}
        if candidate_names is not None
        else pose_names
    )
    target_pose_present = any(
        _same_instance_name(selected_instance, name) for name in pose_names
    )

    metadata: dict[str, Any] = {
        "enabled": bool(enabled),
        "target_instance": selected_instance,
        "target_category": target_category,
        "target_aabb_center": target.get("object_aabb_center"),
        "target_pose_present": target_pose_present,
        "removed_object_names": [],
        "removed_object_count": 0,
    }
    if not enabled:
        metadata["reason"] = "disabled"
        return metadata
    if not selected_instance or not target_category:
        metadata["reason"] = "missing_target_identity_or_category"
        return metadata

    # Never delete an object referenced by the task or by the interaction plan,
    # even if a malformed category label happens to collide with the target.
    protected_names = {
        selected_instance,
        *[str(name) for name in (episode_spec.task_relevant_objects or [])],
        *[
            str(row.get("object_name"))
            for row in (interactive_nav.get("interactions") or [])
            if isinstance(row, Mapping) and row.get("object_name")
        ],
        *[
            str(name)
            for name in (episode_spec.scene_modifications.added_objects or {})
        ],
    }
    existing_removed = {
        str(name) for name in episode_spec.scene_modifications.removed_objects
    }

    def is_protected(name: str) -> bool:
        return any(_same_instance_name(name, protected) for protected in protected_names)

    removed = sorted(
        name
        for name in candidate_names
        if not is_protected(name)
        and not any(
            _same_instance_name(name, existing) for existing in existing_removed
        )
        and _instance_category(name) == target_category
    )
    if not removed:
        metadata["reason"] = "no_same_category_distractors"
        return metadata

    episode_spec.scene_modifications.removed_objects = sorted(
        existing_removed | set(removed)
    )
    # Avoid replay warnings and prevent removed objects from contributing to
    # workspace-center calculations before the compiled scene is inspected.
    all_removed = set(episode_spec.scene_modifications.removed_objects)
    episode_spec.scene_modifications.object_poses = {
        name: pose
        for name, pose in object_poses.items()
        if not any(
            _same_instance_name(str(name), removed_name)
            for removed_name in all_removed
        )
    }
    metadata["removed_object_names"] = removed
    metadata["removed_object_count"] = len(removed)
    metadata["reason"] = "removed_same_category_distractors"
    return metadata
