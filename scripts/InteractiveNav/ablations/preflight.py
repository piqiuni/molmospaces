"""Fail before creating ablation output when the external model is absent."""

import socket
from urllib.parse import urlsplit


def check_model_endpoints(config):
    endpoints = config.get("model_endpoints") or []
    if not endpoints:
        raise ValueError("at least one model endpoint is required")
    for endpoint in endpoints:
        parsed = urlsplit(str(endpoint))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"invalid model endpoint: {endpoint}")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((parsed.hostname, port), timeout=5):
                pass
        except OSError as exc:
            raise RuntimeError(f"model endpoint is unreachable: {endpoint}") from exc
