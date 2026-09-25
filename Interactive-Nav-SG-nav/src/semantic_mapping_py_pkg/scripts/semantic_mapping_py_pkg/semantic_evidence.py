"""Stateless detector/M1 evidence rules, separate from graph mutation.

Room topology and persistence must use the same accepted evidence. Transport
retries do not revoke identity; an explicit negative semantic result can.
"""

from __future__ import annotations

import math

from .geometry_utils import normalize_label
from .graph_rules import PORTAL_LABELS


def _portal_geometry_plausible(size):
    """Reject degenerate RGB-D portal boxes before they become topology."""

    try:
        values = [abs(float(value)) for value in list(size or [])[:3]]
    except (TypeError, ValueError):
        return False
    if len(values) < 3 or not all(math.isfinite(value) for value in values):
        return False
    horizontal = max(values[0], values[1])
    return bool(
        horizontal >= 0.25
        and values[2] >= 0.90
        and horizontal * values[2] >= 0.35
    )


def _portal_attrs_geometry_plausible(attrs):
    attrs = attrs or {}
    for key in (
        "viz_aabb_size",
        "interaction_reference_aabb_size",
        "interaction_reference_obb_size",
    ):
        if key in attrs and attrs.get(key):
            return _portal_geometry_plausible(attrs.get(key))
    return True


def _accepted_semantic_evidence(attrs):
    """Transport progress is not a revocation of the last semantic result."""
    if str(attrs.get("attribute_status") or "").casefold() in {"pending", "failed", "stale"}:
        accepted = attrs.get("attribute_last_ready")
        if isinstance(accepted, dict):
            return accepted
    return attrs


def _has_m1_portal_confirmation(attrs):
    """Return whether M1 positively identified a detector portal as a door."""

    # A detector portal is not necessarily a door.  M1 can correctly reject
    # a wall panel, cabinet face, or other flat structure while the detector
    # keeps its provisional ``portal`` label.  Such a result must never create
    # a synthetic room behind it.
    evidence = _accepted_semantic_evidence(attrs)
    if str(evidence.get("attribute_status") or "").casefold() != "ready":
        return False
    try:
        confidence = float(evidence.get("attribute_confidence", 0.0))
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(confidence) or not 0.5 <= confidence <= 1.0:
        return False
    if str(evidence.get("mllm_interaction_class") or "").casefold() != "portal":
        return False
    if not _portal_attrs_geometry_plausible(attrs):
        return False
    observed_name = normalize_label(evidence.get("m1_observed_object_name"))
    # A portal class with a contradictory concrete name (for example
    # ``thermostat``) is not a confirmed door, even if the detector proposed
    # portal and the crop contains a flat, door-like shape.  M1's concrete
    # identity is the final semantic check for topology.
    return observed_name in PORTAL_LABELS


def persistence_action(node_type, observation_count, attributes):
    """Return latch/clear/keep; graph state is mutated only by its owner."""
    if node_type not in {"portal", "container"}:
        return "keep"
    evidence = _accepted_semantic_evidence(attributes)
    status = str(evidence.get("attribute_status") or "").casefold()
    confidence = float(evidence.get("attribute_confidence", 0.0) or 0.0)
    name = normalize_label(evidence.get("m1_observed_object_name"))
    pending = bool(evidence.get("m1_refrigerator_pending_confirmation", False))
    confirmed = bool(status == "ready" and confidence >= 0.5
                     and (name or evidence.get("mllm_interaction_class")) and not pending)
    if node_type == "portal":
        confirmed = _has_m1_portal_confirmation(attributes)
    if observation_count >= 2 and confirmed:
        return "latch"
    return "clear" if node_type == "portal" else "keep"
