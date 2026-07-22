from __future__ import annotations

import os
from pathlib import Path
import shlex

from .client import MLLMClientConfig


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    candidates = [Path(path).expanduser()] if path else []
    if not candidates:
        current = Path(__file__).resolve()
        candidates = [parent / ".env" for parent in (current, *current.parents)]
        candidates.append(Path.cwd() / ".env")
    selected = next((candidate for candidate in candidates if candidate.is_file()), None)
    if selected is None:
        return None
    for raw_line in selected.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        try:
            tokens = shlex.split(value.strip(), comments=True, posix=True)
            value = tokens[0] if tokens else ""
        except ValueError:
            value = value.strip().strip("\"'")
        if key and (override or key not in os.environ):
            os.environ[key] = value
    return selected


def client_config_from_env(
    *,
    model: str | None = None,
    metrics_path: str | None = None,
) -> MLLMClientConfig:
    return MLLMClientConfig(
        mode=os.environ.get("SEMANTIC_MODEL_MODE", "disabled"),
        endpoint=os.environ.get("SEMANTIC_MODEL_ENDPOINT", ""),
        api_key_env=os.environ.get("SEMANTIC_MODEL_API_KEY_ENV", "OPENAI_API_KEY"),
        model=model or os.environ.get("SEMANTIC_MODEL_NAME", ""),
        protocol=os.environ.get("SEMANTIC_MODEL_PROTOCOL", "openai_responses"),
        command=os.environ.get("SEMANTIC_MODEL_COMMAND", ""),
        timeout_s=float(os.environ.get("SEMANTIC_MODEL_TIMEOUT_S", "20") or 20.0),
        metrics_path=metrics_path
        if metrics_path is not None
        else os.environ.get("SEMANTIC_MODEL_METRICS_PATH", ""),
    )
