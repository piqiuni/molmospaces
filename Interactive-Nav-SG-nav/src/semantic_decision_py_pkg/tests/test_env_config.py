from __future__ import annotations

import os

from semantic_decision_py_pkg.env_config import apply_model_env_overrides, load_env_file
from semantic_mllm_py_pkg.env import client_config_from_env


def test_load_env_file_does_not_override_existing_values(tmp_path, monkeypatch) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "# comment\nexport SEMANTIC_MODEL_MODE=mock\nNEW_VALUE='hello world'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SEMANTIC_MODEL_MODE", "disabled")

    assert load_env_file(path) == path
    assert os.environ["SEMANTIC_MODEL_MODE"] == "disabled"
    assert os.environ["NEW_VALUE"] == "hello world"


def test_explicit_env_file_can_replace_inherited_credential(tmp_path, monkeypatch) -> None:
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=expected-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "stale-key")

    assert load_env_file(path, override=True) == path
    assert os.environ["OPENAI_API_KEY"] == "expected-key"


def test_model_overrides_are_typed_and_explicit(monkeypatch) -> None:
    monkeypatch.setenv("SEMANTIC_MODEL_MODE", "http")
    monkeypatch.setenv("SEMANTIC_MODEL_NAME", "model-x")
    monkeypatch.setenv("SEMANTIC_MODEL_TIMEOUT_S", "12.5")

    config = apply_model_env_overrides({"mode": "disabled", "timeout_s": 20.0})

    assert config["mode"] == "http"
    assert config["model"] == "model-x"
    assert config["timeout_s"] == 12.5


def test_m2_overrides_take_precedence_without_changing_global_environment(monkeypatch) -> None:
    monkeypatch.setenv("SEMANTIC_MODEL_NAME", "shared-vision-model")
    monkeypatch.setenv("SEMANTIC_MODEL_TIMEOUT_S", "8")
    monkeypatch.setenv("SEMANTIC_M2_MODEL_NAME", "strong-text-model")
    monkeypatch.setenv("SEMANTIC_M2_TIMEOUT_S", "30")
    monkeypatch.setenv("SEMANTIC_M2_TIMEOUT_RETRY_COUNT", "1")
    monkeypatch.setenv("SEMANTIC_M2_TIMEOUT_RETRY_BACKOFF_S", "0.5")
    monkeypatch.setenv("SEMANTIC_M2_REASONING_EFFORT", "medium")

    config = apply_model_env_overrides({})

    assert config["model"] == "strong-text-model"
    assert config["timeout_s"] == 30.0
    assert config["timeout_retry_count"] == 1
    assert config["timeout_retry_backoff_s"] == 0.5
    assert config["selection_reasoning_effort"] == "medium"
    assert os.environ["SEMANTIC_MODEL_NAME"] == "shared-vision-model"
    assert os.environ["SEMANTIC_MODEL_TIMEOUT_S"] == "8"
    shared_client_config = client_config_from_env()
    assert shared_client_config.model == "shared-vision-model"
    assert shared_client_config.timeout_s == 8.0


def test_invalid_m2_numeric_override_is_ignored(monkeypatch) -> None:
    monkeypatch.setenv("SEMANTIC_M2_TIMEOUT_S", "not-a-number")
    monkeypatch.setenv("SEMANTIC_M2_TIMEOUT_RETRY_COUNT", "not-an-int")

    config = apply_model_env_overrides(
        {"timeout_s": 20.0, "timeout_retry_count": 0}
    )

    assert config["timeout_s"] == 20.0
    assert config["timeout_retry_count"] == 0
