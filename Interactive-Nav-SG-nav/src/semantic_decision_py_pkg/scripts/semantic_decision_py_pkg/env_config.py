from __future__ import annotations

import os
from pathlib import Path
import shlex
from typing import Mapping


def _candidate_env_paths(explicit_path: str | None = None) -> list[Path]:
    if explicit_path:
        return [Path(explicit_path).expanduser()]
    current = Path(__file__).resolve()
    return [parent / ".env" for parent in [current, *current.parents]] + [Path.cwd() / ".env"]


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Load a small dotenv-compatible file without requiring python-dotenv."""
    selected = next(
        (candidate for candidate in _candidate_env_paths(str(path) if path else None) if candidate.is_file()),
        None,
    )
    if selected is None:
        return None
    for raw_line in selected.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        try:
            tokens = shlex.split(value, comments=True, posix=True)
            value = tokens[0] if tokens else ""
        except ValueError:
            value = value.strip('"\'')
        if override or key not in os.environ:
            os.environ[key] = value
    return selected


def apply_model_env_overrides(
    model_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Merge global model settings, then M2-only settings, over ROS parameters.

    ``SEMANTIC_MODEL_*`` remains the shared default for M1/M2/M3.  An optional
    ``SEMANTIC_M2_*`` value has higher precedence here and is consumed only by
    the decision node, allowing a stronger/slower text model without changing
    the visual M1 request budget.
    """
    result = dict(model_config or {})
    global_env_map = {
        "mode": "SEMANTIC_MODEL_MODE",
        "command": "SEMANTIC_MODEL_COMMAND",
        "endpoint": "SEMANTIC_MODEL_ENDPOINT",
        "api_key_env": "SEMANTIC_MODEL_API_KEY_ENV",
        "model": "SEMANTIC_MODEL_NAME",
        "protocol": "SEMANTIC_MODEL_PROTOCOL",
        "timeout_s": "SEMANTIC_MODEL_TIMEOUT_S",
        "temperature": "SEMANTIC_MODEL_TEMPERATURE",
        "max_tokens": "SEMANTIC_MODEL_MAX_TOKENS",
        "reasoning_effort": "SEMANTIC_MODEL_REASONING_EFFORT",
        "image_detail": "SEMANTIC_MODEL_IMAGE_DETAIL",
        "metrics_path": "SEMANTIC_MODEL_METRICS_PATH",
    }
    m2_env_map = {
        "mode": "SEMANTIC_M2_MODE",
        "command": "SEMANTIC_M2_COMMAND",
        "endpoint": "SEMANTIC_M2_ENDPOINT",
        "api_key_env": "SEMANTIC_M2_API_KEY_ENV",
        "model": "SEMANTIC_M2_MODEL_NAME",
        "protocol": "SEMANTIC_M2_PROTOCOL",
        "timeout_s": "SEMANTIC_M2_TIMEOUT_S",
        "temperature": "SEMANTIC_M2_TEMPERATURE",
        "max_tokens": "SEMANTIC_M2_MAX_TOKENS",
        "selection_reasoning_effort": "SEMANTIC_M2_REASONING_EFFORT",
        "image_detail": "SEMANTIC_M2_IMAGE_DETAIL",
        "metrics_path": "SEMANTIC_M2_METRICS_PATH",
        "timeout_retry_count": "SEMANTIC_M2_TIMEOUT_RETRY_COUNT",
        "timeout_retry_backoff_s": "SEMANTIC_M2_TIMEOUT_RETRY_BACKOFF_S",
        "context_profile": "SEMANTIC_M2_CONTEXT_PROFILE",
        "prompt_file": "SEMANTIC_M2_PROMPT_FILE",
        "recent_decision_limit": "SEMANTIC_M2_RECENT_DECISION_LIMIT",
        "candidate_pool_mode": "SEMANTIC_M2_CANDIDATE_POOL_MODE",
        "candidate_top_k": "SEMANTIC_M2_CANDIDATE_TOP_K",
        "candidate_max_frontier_candidates": "SEMANTIC_M2_CANDIDATE_MAX_FRONTIER_CANDIDATES",
    }

    def apply(env_map: Mapping[str, str]) -> None:
        for config_key, env_key in env_map.items():
            value = os.environ.get(env_key)
            if value is None or value == "":
                continue
            if config_key in {"timeout_s", "temperature", "timeout_retry_backoff_s"}:
                try:
                    result[config_key] = float(value)
                except ValueError:
                    continue
            elif config_key in {
                "max_tokens", "timeout_retry_count", "recent_decision_limit",
                "candidate_top_k", "candidate_max_frontier_candidates",
            }:
                try:
                    result[config_key] = int(value)
                except ValueError:
                    if config_key in {"recent_decision_limit", "candidate_top_k", "candidate_max_frontier_candidates"}:
                        raise ValueError(f"Invalid integer for {env_key}") from None
                    continue
            else:
                result[config_key] = value

    apply(global_env_map)
    apply(m2_env_map)
    return result
