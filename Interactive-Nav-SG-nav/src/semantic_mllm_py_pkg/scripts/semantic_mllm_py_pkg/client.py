from __future__ import annotations

from dataclasses import dataclass, field, replace
import base64
import fcntl
import json
import mimetypes
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time
from typing import Any, Iterable
from urllib import error as urllib_error
from urllib import request


@dataclass
class MLLMClientConfig:
    mode: str = "disabled"
    endpoint: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "qwen3.6-35b-a3b"
    protocol: str = "openai_chat"
    command: str = ""
    timeout_s: float = 20.0
    temperature: float = 0.0
    max_tokens: int = 384
    reasoning_effort: str = "off"
    image_detail: str = "low"
    metrics_path: str = ""


@dataclass
class MLLMResponse:
    payload: dict[str, Any] | None
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    error: str = ""
    raw_text: str = ""
    raw_http_response: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def tps(self) -> float:
        if self.latency_s <= 0.0:
            return 0.0
        tokens = self.completion_tokens
        if tokens <= 0 and self.raw_text:
            tokens = max(1, len(self.raw_text) // 4)
        return float(tokens) / self.latency_s

    def metrics(self) -> dict[str, Any]:
        visible_output_tokens = max(0, self.completion_tokens - self.reasoning_tokens)
        return {
            "latency_s": self.latency_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "visible_output_tokens": visible_output_tokens,
            "total_tokens": self.total_tokens,
            "tps": self.tps,
            "visible_output_tps": (
                float(visible_output_tokens) / self.latency_s
                if self.latency_s > 0.0
                else 0.0
            ),
            "error": self.error,
            "raw_text_chars": len(self.raw_text),
            "raw_text": self.raw_text,
        }


class MLLMClient:
    """Small dependency-free client shared by all MLLM roles."""

    def __init__(self, config: MLLMClientConfig | None = None) -> None:
        self.config = config or MLLMClientConfig()
        self._metrics_lock = threading.Lock()

    def request_json(
        self,
        *,
        role: str,
        instruction: str,
        context: dict[str, Any],
        images: Iterable[str] | None = None,
        response_schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        max_tokens: int | None = None,
        metrics_context: dict[str, Any] | None = None,
    ) -> MLLMResponse:
        started = time.perf_counter()
        images = list(images or [])
        config = self.config
        overrides: dict[str, Any] = {}
        if timeout_s is not None:
            overrides["timeout_s"] = max(0.1, float(timeout_s))
        if max_tokens is not None:
            overrides["max_tokens"] = max(1, int(max_tokens))
        if overrides:
            config = replace(config, **overrides)
        try:
            mode = str(config.mode or "disabled").casefold()
            if mode == "disabled":
                raise RuntimeError("MLLM client is disabled")
            if mode == "mock":
                payload = self._mock_response(role, context)
                response = MLLMResponse(payload=payload, latency_s=time.perf_counter() - started)
            elif mode == "command":
                payload, raw_text = self._request_command(
                    role, instruction, context, images, response_schema, config
                )
                response = MLLMResponse(payload=payload, latency_s=time.perf_counter() - started, raw_text=raw_text)
            elif mode == "http":
                response = self._request_http(
                    role, instruction, context, images, response_schema, started, config
                )
            else:
                raise ValueError(f"unsupported MLLM mode: {config.mode}")
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = f"HTTP {exc.code}: {body[:1000]}" if body else str(exc)
            response = MLLMResponse(
                payload=None,
                latency_s=time.perf_counter() - started,
                error=message,
                raw_http_response=body,
            )
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, TimeoutError) as exc:
            response = MLLMResponse(payload=None, latency_s=time.perf_counter() - started, error=str(exc))
        self._record_metrics(role, response, config, metrics_context)
        return response

    def _request_http(
        self,
        role: str,
        instruction: str,
        context: dict[str, Any],
        images: list[str],
        response_schema: dict[str, Any] | None,
        started: float,
        config: MLLMClientConfig,
    ) -> MLLMResponse:
        protocol = str(config.protocol or "openai_chat").casefold()
        request_instruction = self._instruction_for_request(instruction, config)
        if protocol in {"generic", "interactive_navigation"}:
            body_payload: dict[str, Any] = {
                "schema_version": 1,
                "role": role,
                "instruction": request_instruction,
                "context": context,
                "images": images,
                "response_schema": response_schema or {},
            }
            if config.model:
                body_payload["model"] = config.model
        elif protocol in {"openai_responses", "responses"}:
            content: list[dict[str, Any]] = [
                {
                    "type": "input_text",
                    "text": request_instruction + "\n" + json.dumps(context, ensure_ascii=False),
                }
            ]
            for image in images:
                image_item = {
                    "type": "input_image",
                    "image_url": self._image_url(image),
                }
                if config.image_detail:
                    image_item["detail"] = config.image_detail
                content.append(image_item)
            body_payload = {
                "model": config.model,
                "max_output_tokens": config.max_tokens,
                "input": [{"role": "user", "content": content}],
            }
            reasoning_effort = self._reasoning_effort_for_request(config)
            if reasoning_effort:
                body_payload["reasoning"] = {"effort": reasoning_effort}
            if self._thinking_disabled(config):
                body_payload["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            content: list[dict[str, Any]] = [{"type": "text", "text": request_instruction + "\n" + json.dumps(context, ensure_ascii=False)}]
            for image in images:
                content.append({"type": "image_url", "image_url": {"url": self._image_url(image)}})
            body_payload = {
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "Return only a valid JSON object."},
                    {"role": "user", "content": content},
                ],
            }
            reasoning_effort = self._reasoning_effort_for_request(config)
            if reasoning_effort:
                body_payload["reasoning_effort"] = reasoning_effort
            if self._thinking_disabled(config):
                body_payload["enable_thinking"] = False
        body = json.dumps(body_payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(config.api_key_env, "") if config.api_key_env else ""
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        endpoint = self._resolved_endpoint(protocol, endpoint=config.endpoint)
        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        with request.urlopen(req, timeout=config.timeout_s) as response_obj:
            raw = response_obj.read().decode("utf-8")
        envelope = json.loads(raw)
        usage = envelope.get("usage") or {}
        output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
        raw_text = self._extract_text(envelope)
        if not raw_text:
            raise ValueError("MLLM response contained no output text")
        try:
            payload = self._parse_json(raw_text)
            parse_error = ""
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            payload = None
            parse_error = f"invalid JSON response: {exc}"
        return MLLMResponse(
            payload=payload,
            latency_s=time.perf_counter() - started,
            prompt_tokens=int(
                usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
            ),
            completion_tokens=int(
                usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            ),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
            reasoning_tokens=int(
                output_details.get("reasoning_tokens", output_details.get("reasoning", 0)) or 0
            ),
            raw_text=raw_text,
            raw_http_response=raw,
            usage=dict(usage),
            error=parse_error,
        )

    def _request_command(
        self,
        role: str,
        instruction: str,
        context: dict[str, Any],
        images: list[str],
        response_schema: dict[str, Any] | None,
        config: MLLMClientConfig,
    ) -> tuple[dict[str, Any], str]:
        command = shlex.split(config.command)
        if not command:
            raise ValueError("MLLM command is empty")
        payload = {
            "role": role,
            "instruction": instruction,
            "context": context,
            "images": images,
            "response_schema": response_schema or {},
            "model": config.model,
        }
        completed = subprocess.run(
            command,
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=config.timeout_s,
            check=True,
        )
        raw_text = completed.stdout.strip()
        return self._parse_json(raw_text), raw_text

    def _resolved_endpoint(self, protocol: str, endpoint: str | None = None) -> str:
        endpoint = str(self.config.endpoint if endpoint is None else endpoint or "").rstrip("/")
        if not endpoint:
            raise ValueError("MLLM endpoint is empty")
        if protocol in {"openai_chat", "chat_completions", "openai"} and endpoint.endswith(
            "/v1"
        ):
            return endpoint + "/chat/completions"
        if protocol in {"openai_responses", "responses"} and endpoint.endswith("/v1"):
            return endpoint + "/responses"
        return endpoint

    def _reasoning_effort_for_request(
        self, config: MLLMClientConfig | None = None
    ) -> str:
        config = config or self.config
        value = str(config.reasoning_effort or "").strip()
        if self._thinking_disabled(config):
            return "none"
        return value

    def _thinking_disabled(self, config: MLLMClientConfig | None = None) -> bool:
        config = config or self.config
        value = str(config.reasoning_effort or "").strip()
        return value.casefold() in {"off", "none", "false", "0", "disabled"}

    def _instruction_for_request(
        self, instruction: str, config: MLLMClientConfig | None = None
    ) -> str:
        if not self._thinking_disabled(config) or "/no_think" in instruction:
            return instruction
        return instruction.rstrip() + "\n/no_think"

    def _mock_response(self, role: str, context: dict[str, Any]) -> dict[str, Any]:
        if role == "subgoal_selection":
            candidates = context.get("candidates") or []
            return {"candidate_id": str(candidates[0].get("candidate_id"))} if candidates else {}
        if role == "attribute_inference":
            return {
                "object_id": str(context.get("object_id") or "unknown"),
                "interactable": False,
                "interaction_class": "unknown",
                "coarse_state": "unknown",
                "interaction_parts": [],
                "confidence": 0.0,
            }
        if role == "room_attribute_inference":
            return {
                "room_id": context.get("room_id"),
                "room_attribute": "unknown",
                "confidence": 0.0,
                "evidence_object_ids": [],
            }
        if role == "skill_planning":
            expected_type = str(context.get("expected_target_type") or "unknown")
            if expected_type == "drawer_container":
                return {
                    "target_type": "drawer_container",
                    "action": "scan",
                    "operation_method": "pull",
                    "open_regions": [
                        {"center": [0.5, 0.25], "confidence": 0.5},
                        {"center": [0.5, 0.75], "confidence": 0.5},
                    ],
                    "confidence": 0.5,
                    "reason": "mock drawer regions",
                }
            return {
                "target_type": "door" if expected_type == "door" else expected_type,
                "action": str(context.get("requested_action") or "open"),
                "operation_method": "hinged_unknown" if expected_type == "door" else "unknown",
                "open_regions": [],
                "confidence": 0.5,
                "reason": "mock operation plan",
            }
        return {"success": False, "confidence": 0.0, "reason": "mock"}

    @staticmethod
    def _parse_json(raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0]
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("MLLM response must be a JSON object")
        return value

    @staticmethod
    def _extract_text(envelope: dict[str, Any]) -> str:
        if isinstance(envelope.get("output_text"), str):
            return str(envelope["output_text"])
        output = envelope.get("output") or []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            texts = []
            for content_item in item.get("content") or []:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") in {"output_text", "text"}:
                    texts.append(str(content_item.get("text") or ""))
            if texts:
                return "".join(texts)
        choices = envelope.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return str(content)

    @staticmethod
    def _image_url(image: str) -> str:
        if image.startswith("data:") or image.startswith("http://") or image.startswith("https://"):
            return image
        path = Path(image).expanduser()
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _record_metrics(
        self,
        role: str,
        response: MLLMResponse,
        config: MLLMClientConfig,
        metrics_context: dict[str, Any] | None,
    ) -> None:
        if not config.metrics_path:
            return
        record = {
            "timestamp": time.time(),
            "role": role,
            "model": config.model,
            "timeout_s": config.timeout_s,
            "max_output_tokens": config.max_tokens,
            "protocol": config.protocol,
            "reasoning_effort": config.reasoning_effort,
            **(metrics_context or {}),
            **response.metrics(),
        }
        path = Path(config.metrics_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._metrics_lock, path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
