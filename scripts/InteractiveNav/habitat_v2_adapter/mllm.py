"""Thin external bridge to the existing InteractiveNav Module-2 client.

The implementation intentionally reuses ``ModelPolicyClient`` instead of
reimplementing prompts, response schemas, retries, or MLLM transport.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable


def _source_paths() -> tuple[Path, Path]:
    repository_root = Path(__file__).resolve().parents[3]
    source_root = repository_root / "Interactive-Nav-SG-nav" / "src"
    return (
        source_root / "semantic_decision_py_pkg" / "scripts",
        source_root / "semantic_mllm_py_pkg" / "scripts",
    )


def _ensure_interactive_nav_imports() -> None:
    for source_path in reversed(_source_paths()):
        source_text = str(source_path)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)


def build_model_policy(
    *,
    endpoint: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
    metrics_path: str,
):
    """Construct the unmodified current InteractiveNav Module-2 policy."""

    _ensure_interactive_nav_imports()
    from semantic_decision_py_pkg.model_policy import (  # type: ignore[import-not-found]
        ModelPolicyClient,
        ModelPolicyConfig,
    )

    return ModelPolicyClient(
        ModelPolicyConfig(
            mode="http",
            endpoint=endpoint,
            api_key_env="SEMANTIC_MODEL_API_KEY",
            model=model,
            protocol="openai_chat",
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            reasoning_effort="off",
            metrics_path=metrics_path,
        )
    )


def build_vision_client(
    *,
    endpoint: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
    metrics_path: str,
):
    """Create the existing shared MLLM transport for public RGB observations.

    Module-2 remains the InteractiveNav subgoal selector.  This helper uses the
    same current InteractiveNav transport to turn a public RGB observation into
    a category-presence signal; it does not create interaction attributes or
    actions.
    """

    _ensure_interactive_nav_imports()
    from semantic_mllm_py_pkg.client import (  # type: ignore[import-not-found]
        MLLMClient,
        MLLMClientConfig,
    )

    return MLLMClient(
        MLLMClientConfig(
            mode="http",
            endpoint=endpoint,
            api_key_env="SEMANTIC_MODEL_API_KEY",
            model=model,
            protocol="openai_chat",
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            reasoning_effort="off",
            metrics_path=metrics_path,
        )
    )


def build_objectgoal_stop_verifier(client: Any, *, min_confidence: float):
    """Build the original InteractiveNav navigation-only Module-3 verifier."""

    _ensure_interactive_nav_imports()
    from semantic_decision_py_pkg.objectgoal_stop_verifier import ObjectGoalStopVerifier

    return ObjectGoalStopVerifier(client, min_confidence=min_confidence)


def build_behavior_candidates(records: Iterable[dict[str, Any]]):
    """Create only navigation candidates accepted by the existing M2 policy.

    ``INTERACT`` is forbidden at the external boundary and is checked before
    every model call.  The current M2 schema then makes an interaction command
    impossible to select or execute.
    """

    _ensure_interactive_nav_imports()
    from semantic_decision_py_pkg.behavior_candidates import (  # type: ignore[import-not-found]
        BEHAVIOR_EXPLORE,
        BEHAVIOR_NAVIGATE,
        BehaviorCandidate,
    )

    accepted = {BEHAVIOR_EXPLORE, BEHAVIOR_NAVIGATE}
    candidates = []
    for record in records:
        behavior_type = str(record["behavior_type"]).upper()
        if behavior_type not in accepted:
            raise ValueError(f"navigation-only adapter rejected {behavior_type!r}")
        candidate = BehaviorCandidate(
            candidate_id=str(record["candidate_id"]),
            behavior_type=behavior_type,
            source="habitat_objectnav_v2_adapter",
            target_id=str(record["target_id"]),
            target_name=str(record["target_name"]),
            goal_xyyaw=[float(value) for value in record["goal_xyyaw"]],
            interaction_command=None,
            features={str(key): float(value) for key, value in record["features"].items()},
            metadata=dict(record["metadata"]),
        )
        if candidate.interaction_command is not None:
            raise AssertionError("external policy must never send interaction commands")
        candidates.append(candidate)
    return candidates


def candidate_history_key_for(candidate: Any, graph: dict[str, Any], *, region_size_m: float = 1.0) -> str:
    """Use the unmodified Module-2 region key for adapter-side public history."""

    _ensure_interactive_nav_imports()
    from semantic_decision_py_pkg.candidate_curator import (  # type: ignore[import-not-found]
        candidate_history_key,
    )

    return str(candidate_history_key(candidate, graph, region_size_m=region_size_m))


def local_model_environment(endpoint: str) -> None:
    """Set only a non-secret local default used by the vLLM service."""

    if endpoint.startswith(("http://127.0.0.1", "http://localhost")):
        os.environ.setdefault("SEMANTIC_MODEL_API_KEY", "local")
