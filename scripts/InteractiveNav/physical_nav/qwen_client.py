#!/usr/bin/env python3
"""Small OpenAI-compatible client for a Qwen service reached through SSH."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any


class QwenClient:
    def __init__(self, base_url: str = "http://127.0.0.1:18080/v1", model: str = "qwen3.6-35b-a3b-fp8", timeout_s: float = 30.0, api_key: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.api_key = api_key or os.environ.get("SEMANTIC_MODEL_API_KEY", "")

    def chat(self, prompt: str, *, image_data_url: str | None = None, max_tokens: int = 256, response_format: dict[str, Any] | None = None, timeout_s: float | None = None) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_data_url:
            content.append({"type": "image_url", "image_url": {"url": image_data_url}})
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}
        if response_format:
            payload["response_format"] = dict(response_format)
        headers = {"Content-Type": "application/json"}
        if self.api_key: headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(), headers=headers)
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s if timeout_s is None else min(self.timeout_s, max(0.01, timeout_s))) as response:
                value = json.loads(response.read().decode())
            value["latency_s"] = time.monotonic() - started
            return value
        except Exception as exc:
            return {"error": str(exc), "latency_s": time.monotonic() - started}
