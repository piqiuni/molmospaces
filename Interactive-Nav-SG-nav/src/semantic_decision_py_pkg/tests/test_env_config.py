from __future__ import annotations

import os

from semantic_decision_py_pkg.env_config import apply_model_env_overrides, load_env_file


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


def test_model_overrides_are_typed_and_explicit(monkeypatch) -> None:
    monkeypatch.setenv("SEMANTIC_MODEL_MODE", "http")
    monkeypatch.setenv("SEMANTIC_MODEL_NAME", "model-x")
    monkeypatch.setenv("SEMANTIC_MODEL_TIMEOUT_S", "12.5")

    config = apply_model_env_overrides({"mode": "disabled", "timeout_s": 20.0})

    assert config["mode"] == "http"
    assert config["model"] == "model-x"
    assert config["timeout_s"] == 12.5
